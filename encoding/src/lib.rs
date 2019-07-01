mod constants;
mod function;
mod instructions;
mod state;
mod value_type;

pub use crate::{
	constants::Constants,
	function::Function,
	instructions::{stack_depth, Instruction},
	state::State,
	value_type::value_type_sort,
};

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

	ctx.from_bool(true).and(&[
		&stack_pointers_equal,
		&stacks_equal,
		&n_locals_equal,
		&locals_equal,
	])
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
	I32(i32),
	I64(i64),
	F32(f32),
	F64(f64),
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;

	#[test]
	fn stack_depth_test() {
		let program = &[];
		assert_eq!(stack_depth(program), 0);

		let program = &[I32Add];
		assert_eq!(stack_depth(program), 2);

		let program = &[I32Const(1), I32Add];
		assert_eq!(stack_depth(program), 1);

		let program = &[I32Const(1), I32Const(1), I32Const(1), I32Add];
		assert_eq!(stack_depth(program), 0);
	}
}
