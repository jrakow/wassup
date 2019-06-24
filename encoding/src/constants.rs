use crate::instructions::*;
use parity_wasm::elements::ValueType;
use z3::*;

/// Encoded types and expressions that are independent of a specific instruction sequence.
pub struct Constants<'ctx> {
	pub ctx: &'ctx Context,
	/// Uninterpreted initial stack values
	pub initial_stack: Vec<Ast<'ctx>>,
	/// Type of the initial stack
	pub initial_stack_types: Vec<ValueType>,
	/// Arguments == Values of the first locals
	pub params: Vec<Ast<'ctx>>,
}

impl<'ctx> Constants<'ctx> {
	pub fn new(ctx: &'ctx Context, n_params: usize, initial_stack_types: &[ValueType]) -> Self {
		let word_sort = ctx.bitvector_sort(32);
		let initial_stack: Vec<_> = (0..initial_stack_types.len())
			.map(|_| ctx.fresh_const("initial_stack", &word_sort))
			.collect();
		let params: Vec<_> = (0..n_params)
			.map(|_| ctx.fresh_const("param", &word_sort))
			.collect();

		Self {
			ctx,
			initial_stack,
			initial_stack_types: initial_stack_types.to_vec(),
			params,
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

	/// Set the function arguments
	///
	/// This is useful for evaluating operations on arguments, compared to doing operations for all possible arguments.
	pub fn set_params(&self, solver: &Solver, params: &[u32]) {
		for (i, v) in params.iter().enumerate() {
			let v = self.ctx.from_u32(*v).int2bv(32);

			solver.assert(&self.params[i]._eq(&v));
		}
	}
}
