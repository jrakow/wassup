use crate::instructions::*;
use parity_wasm::elements::ValueType;
use z3::*;

/// Encoded types and expressions that are independent of a specific instruction sequence.
pub struct Constants<'ctx> {
	pub ctx: &'ctx Context,
	/// Uninterpreted initial stack values
	pub initial_stack: Vec<Ast<'ctx>>,
	/// Arguments == Values of the first locals
	pub initial_locals: Vec<Ast<'ctx>>,
}

impl<'ctx> Constants<'ctx> {
	pub fn new(
		ctx: &'ctx Context,
		initial_locals: Vec<Ast<'ctx>>,
		initial_stack_types: &[ValueType],
	) -> Self {
		let word_sort = ctx.bitvector_sort(32);
		let initial_stack: Vec<_> = (0..initial_stack_types.len())
			.map(|_| ctx.fresh_const("initial_stack", &word_sort))
			.collect();

		Self {
			ctx,
			initial_stack,
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
