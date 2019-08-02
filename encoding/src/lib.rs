mod constants;
mod execution;
mod function;
mod instructions;
mod state;

use crate::state::*;
pub use crate::{
	constants::Constants,
	execution::Execution,
	function::Function,
	instructions::{initial_stack_types, Instruction},
	state::State,
};
use parity_wasm::elements::ValueType;
use z3::*;

/// a <= b < c
pub fn in_range<'ctx>(a: &Ast<'ctx>, b: &Ast<'ctx>, c: &Ast<'ctx>) -> Ast<'ctx> {
	a.le(&b).and(&[&b.lt(&c)])
}

/// Whether state `lhs` is equivalent to state `rhs`
///
/// This expression contains quantifiers and cannot be tested directly as Z3 does not support evaluating terms with quantifiers.
pub fn equivalent<'ctx>(
	ctx: &'ctx Context,
	lhs: &EncodedState<'ctx>,
	rhs: &EncodedState<'ctx>,
	n_locals: &Ast<'ctx>,
) -> Ast<'ctx> {
	let stack_pointers_equal = lhs.stack_pointer()._eq(&rhs.stack_pointer());

	let stacks_equal = {
		// for 0 <= n < stack_pointer
		let n = ctx.named_int_const("n");
		let n_in_range = in_range(&ctx.from_u64(0), &n, &lhs.stack_pointer());

		let values_equal = lhs.stack(&n)._eq(&rhs.stack(&n));
		let types_equal = lhs.stack_type(&n)._eq(&rhs.stack_type(&n));

		ctx.forall_const(
			&[&n],
			&n_in_range.implies(&values_equal.and(&[&types_equal])),
		)
	};

	// both states have the same number of locals as they have the same constants
	let locals_equal = {
		let n = ctx.named_int_const("n");
		let n_in_range = in_range(&ctx.from_u64(0), &n, &n_locals);

		let condition = lhs.local(&n)._eq(&rhs.local(&n));

		ctx.forall_const(&[&n], &n_in_range.implies(&condition))
	};

	let left_trapped = lhs.trapped();
	let right_trapped = rhs.trapped();
	let both_trapped = left_trapped.and(&[&right_trapped]);
	let trapped_equal = left_trapped._eq(&right_trapped);

	let states_equal = ctx.from_bool(true).and(&[
		&stack_pointers_equal,
		&stacks_equal,
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
	pub fn encode<'ctx>(
		&self,
		ctx: &'ctx Context,
		value_type_config: ValueTypeConfig,
	) -> Ast<'ctx> {
		match self {
			Value::I32(i) => {
				let size = value_type_config.i32_size;
				if size < 32 {
					assert!((*i as u32) < 1 << size as u32);
				}

				if let Some(i64_size) = value_type_config.i64_size {
					// extend to size of I64
					ctx.from_i32(*i).int2bv(i64_size as u64)
				} else {
					ctx.from_i32(*i).int2bv(size as u64)
				}
			}
			Value::I64(i) => {
				let size = value_type_config.i64_size.unwrap();
				if size < 64 {
					assert!((*i as u64) < 1 << size as u64);
				}

				ctx.from_i64(*i).int2bv(size as u64)
			}
			_ => unimplemented!(),
		}
	}

	pub fn value_type(&self) -> ValueType {
		match self {
			Value::I32(_) => ValueType::I32,
			Value::I64(_) => ValueType::I64,
			_ => unimplemented!(),
		}
	}

	pub fn decode(
		v: &Ast,
		model: &Model,
		value_type: ValueType,
		value_type_config: ValueTypeConfig,
	) -> Self {
		match value_type {
			ValueType::I32 => {
				// only lower part
				let int = v.bvextract(value_type_config.i32_size - 1, 0).bv2int(true);
				let int = model.eval(&int).unwrap();

				Value::I32(int.as_i32().unwrap())
			}
			ValueType::I64 => {
				let int = model.eval(&v.bv2int(true)).unwrap();
				Value::I64(int.as_i64().unwrap())
			}
			_ => unimplemented!(),
		}
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// How to represent a value type
///
/// I32 is always needed, so there is no 64 bit only
pub struct ValueTypeConfig {
	pub i32_size: usize,
	pub i64_size: Option<usize>,
}

impl ValueTypeConfig {
	/// Representation of a value in Z3
	pub fn value_sort<'ctx>(&self, ctx: &'ctx Context) -> Sort<'ctx> {
		ctx.bitvector_sort(self.i64_size.unwrap_or(self.i32_size) as u32)
	}

	/// Representation of a value type in Z3
	pub fn value_type_datatype<'ctx>(&self, ctx: &'ctx Context) -> Datatype<'ctx> {
		let mut builder = DatatypeBuilder::new(ctx).variant("I32", &[]);
		if self.i64_enabled() {
			builder = builder.variant("I64", &[]);
		}

		builder.finish("ValueType")
	}

	pub fn encode_value_type<'ctx>(&self, ctx: &'ctx Context, value_type: ValueType) -> Ast<'ctx> {
		let datatype = self.value_type_datatype(ctx);
		match value_type {
			ValueType::I32 => datatype.variants[0].constructor.apply(&[]),
			ValueType::I64 => datatype.variants[1].constructor.apply(&[]),
			_ => unimplemented!(),
		}
	}

	pub fn decode_value_type<'ctx>(
		&self,
		ctx: &'ctx Context,
		model: &Model<'ctx>,
		i: &Ast<'ctx>,
	) -> ValueType {
		let datatype = self.value_type_datatype(ctx);

		if self.i64_size.is_none() {
			ValueType::I32
		} else {
			if model
				.eval(&datatype.variants[0].tester.apply(&[&i]))
				.unwrap()
				.as_bool()
				.unwrap()
			{
				ValueType::I32
			} else {
				ValueType::I64
			}
		}
	}

	pub fn i64_enabled(&self) -> bool {
		self.i64_size.is_some()
	}

	pub fn i32_wrap_as_i64<'ctx>(&self, ctx: &'ctx Context, i: &Ast<'ctx>) -> Ast<'ctx> {
		if let Some(i64_size) = self.i64_size {
			// fill upper part of word with undefined bits
			let upper_sort = ctx.bitvector_sort((i64_size - self.i32_size) as u32);
			let undefined = ctx.fresh_const("undefined", &upper_sort);

			undefined.concat(i)
		} else {
			i.clone()
		}
	}

	pub fn i64_unwrap_as_i32<'ctx>(&self, i: &Ast<'ctx>) -> Ast<'ctx> {
		if self.i64_size.is_some() {
			i.bvextract(self.i32_size - 1, 0)
		} else {
			i.clone()
		}
	}
}
