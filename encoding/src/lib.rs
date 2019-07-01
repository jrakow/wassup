mod constants;
mod function;
mod instructions;
mod state;

pub use crate::{
	constants::Constants,
	function::Function,
	instructions::{stack_depth, Instruction},
	state::State,
};
use parity_wasm::elements::ValueType;
use z3::*;

/// a <= b < c
pub fn in_range<'ctx>(a: &Ast<'ctx>, b: &Ast<'ctx>, c: &Ast<'ctx>) -> Ast<'ctx> {
	a.le(&b).and(&[&b.lt(&c)])
}

/// Whether state `lhs` at `lhs_pc` is equivalent to state `rhs` at `rhs_pc`
///
/// This expression contains quantifiers and cannot be tested directly as Z3 does not support evaluating terms with quantifiers.
pub fn equivalent<'ctx>(
	lhs: &State<'ctx, '_, '_>,
	lhs_pc: &Ast<'ctx>,
	rhs: &State<'ctx, '_, '_>,
	rhs_pc: &Ast<'ctx>,
) -> Ast<'ctx> {
	let ctx = lhs.ctx;

	let stack_pointers_equal = lhs.stack_pointer(&lhs_pc)._eq(&rhs.stack_pointer(&rhs_pc));

	let stacks_equal = {
		// for 0 <= n < stack_pointer
		let n = ctx.named_int_const("n");
		let n_in_range = in_range(&ctx.from_u64(0), &n, &lhs.stack_pointer(&lhs_pc));

		// lhs-stack(lhs_pc, n) ==  rhs-stack(rhs_pc, n)
		let condition = lhs.stack(&lhs_pc, &n)._eq(&rhs.stack(&rhs_pc, &n));

		ctx.forall_const(&[&n], &n_in_range.implies(&condition))
	};

	// require that both states have the same number of locals and that they are all equal
	// TODO maybe relax this in the future for more optimization potential
	let n_locals_equal = lhs.n_locals()._eq(&rhs.n_locals());
	let locals_equal = {
		let n = ctx.named_int_const("n");
		let n_in_range = in_range(&ctx.from_u64(0), &n, &lhs.n_locals());

		let condition = lhs.local(&lhs_pc, &n)._eq(&rhs.local(&rhs_pc, &n));

		ctx.forall_const(&[&n], &n_in_range.implies(&condition))
	};

	let left_trapped = lhs.trapped(lhs_pc);
	let right_trapped = rhs.trapped(rhs_pc);
	let both_trapped = left_trapped.and(&[&right_trapped]);
	let trapped_equal = left_trapped._eq(&right_trapped);

	let states_equal = ctx.from_bool(true).and(&[
		&stack_pointers_equal,
		&stacks_equal,
		&n_locals_equal,
		&locals_equal,
		&trapped_equal,
	]);

	both_trapped.or(&[&states_equal])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
	I32(i32),
	I64(i64),
	F32(f32),
	F64(f64),
}

impl Value {
	pub fn encode<'ctx>(&self, ctx: &'ctx Context) -> Ast<'ctx> {
		let datatype = value_type(ctx);

		let (variant_index, encoded_value) = match self {
			Value::I32(i) => (0, ctx.from_i32(*i).int2bv(32)),
			Value::I64(i) => (1, ctx.from_i64(*i).int2bv(64)),
			_ => unimplemented!(),
		};

		datatype.variants[variant_index]
			.constructor
			.apply(&[&encoded_value])
	}

	pub fn decode(v: &Ast, ctx: &Context, model: &Model) -> Self {
		let dataype = value_type(ctx);

		let index = dataype
			.variants
			.iter()
			.position(|variant| {
				let active = variant.tester.apply(&[&v]);
				model.eval(&active).unwrap().as_bool().unwrap()
			})
			.unwrap();

		let inner = model
			.eval(
				&dataype.variants[index].accessors[0]
					.apply(&[&v])
					.bv2int(true),
			)
			.unwrap();

		match index {
			0 => Value::I32(inner.as_i32().unwrap()),
			1 => Value::I64(inner.as_i64().unwrap()),
			2 | 3 => unimplemented!(),
			_ => unreachable!(),
		}
	}
}

/// Representation of a value in Z3
pub fn value_type(ctx: &Context) -> Datatype {
	DatatypeBuilder::new(ctx)
		.variant("I32", &[("as_i32", &ctx.bitvector_sort(32))])
		.variant("I64", &[("as_i64", &ctx.bitvector_sort(64))])
		.finish("Value")
}

pub fn value_type_to_index(v: &ValueType) -> usize {
	match v {
		ValueType::I32 => 0,
		ValueType::I64 => 1,
		_ => unimplemented!(),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;
	use Value::*;

	#[test]
	fn stack_depth_test() {
		let program = &[];
		assert_eq!(stack_depth(program), 0);

		let program = &[I32Add];
		assert_eq!(stack_depth(program), 2);

		let program = &[Const(I32(1)), I32Add];
		assert_eq!(stack_depth(program), 1);

		let program = &[Const(I32(1)), Const(I32(1)), Const(I32(1)), I32Add];
		assert_eq!(stack_depth(program), 0);
	}
}
