use crate::instructions::*;
use crate::{value_type, value_type_to_index};
use parity_wasm::elements::ValueType;
use z3::*;

/// Encoded types and expressions that are independent of a specific instruction sequence.
pub struct Constants<'ctx> {
	pub ctx: &'ctx Context,
	/// Values of the initial stack
	pub initial_stack: Vec<Ast<'ctx>>,
	/// Z3 constants used in initial_stack
	pub initial_stack_bounds: Vec<Ast<'ctx>>,
	pub initial_locals: Vec<Ast<'ctx>>,
}

impl<'ctx> Constants<'ctx> {
	pub fn new(
		ctx: &'ctx Context,
		initial_locals: Vec<Ast<'ctx>>,
		initial_stack_types: &[ValueType],
	) -> Self {
		let mut initial_stack = Vec::new();
		let mut initial_stack_bounds = Vec::new();
		for ty in initial_stack_types {
			let datatype = value_type(ctx);
			let sort = match ty {
				ValueType::I32 => ctx.bitvector_sort(32),
				_ => unimplemented!(),
			};
			let inner = ctx.fresh_const("initial_stack", &sort);
			initial_stack_bounds.push(inner.clone());

			let value = datatype.variants[value_type_to_index(ty)]
				.constructor
				.apply(&[&inner]);
			initial_stack.push(value);
		}

		Self {
			ctx,
			initial_stack,
			initial_stack_bounds,
			initial_locals,
		}
	}

	/// How many values the instruction takes from the stack.
	///
	/// This is here so it can be encoded as a function, not as a big expression.
	pub fn stack_pop_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let instruction_datatype = instruction_datatype(self.ctx);
		// build a switch statement from if-then-else
		let mut pop_switch = self.ctx.from_usize(0);
		for i in Instruction::iter_templates() {
			let (pops, _) = i.stack_pop_push_count();

			let variant = &instruction_datatype.variants[i.as_usize()];
			let active = variant.tester.apply(&[&instr]);

			pop_switch = active.ite(&self.ctx.from_usize(pops), &pop_switch);
		}
		pop_switch
	}

	/// How many values the instruction pushes on the stack.
	///
	/// This is here so it can be encoded as a function, not as a big expression.
	pub fn stack_push_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let instruction_datatype = instruction_datatype(self.ctx);

		// build a switch statement from if-then-else
		let mut push_switch = self.ctx.from_usize(0);
		for i in Instruction::iter_templates() {
			let (_, pushs) = i.stack_pop_push_count();

			let variant = &instruction_datatype.variants[i.as_usize()];
			let active = variant.tester.apply(&[&instr]);

			push_switch = active.ite(&self.ctx.from_usize(pushs), &push_switch);
		}
		push_switch
	}
}
