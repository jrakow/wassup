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

impl<'ctx, 'solver> Constants<'ctx> {
	pub fn new(
		ctx: &'ctx Context,
		solver: &Solver<'ctx>,
		n_params: usize,
		initial_stack_types: &[ValueType],
	) -> Self {
		let word_sort = ctx.bitvector_sort(32);
		let initial_stack: Vec<_> = (0..initial_stack_types.len())
			.map(|_| ctx.fresh_const("initial_stack", &word_sort))
			.collect();
		let params: Vec<_> = (0..n_params)
			.map(|_| ctx.fresh_const("param", &word_sort))
			.collect();

		let constants = Constants {
			ctx,
			initial_stack,
			initial_stack_types: initial_stack_types.to_vec(),
			params,
		};

		for ref i in Instruction::iter_templates() {
			let (pops, pushs) = i.stack_pop_push_count();
			solver.assert(
				&constants
					.stack_pop_count(&i.encode(ctx))
					._eq(&ctx.from_usize(pops)),
			);
			solver.assert(
				&constants
					.stack_push_count(&i.encode(ctx))
					._eq(&ctx.from_usize(pushs)),
			);
		}

		constants
	}

	/// How many values the instruction takes from the stack.
	///
	/// This is here so it can be encoded as a function, not as a big expression.
	pub fn stack_pop_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_pop_count_func = self.ctx.func_decl(
			self.ctx.str_sym("stack_pop_count"),
			&[&instruction_sort(self.ctx).0],
			&self.ctx.int_sort(),
		);

		stack_pop_count_func.apply(&[instr])
	}

	/// How many values the instruction pushes on the stack.
	///
	/// This is here so it can be encoded as a function, not as a big expression.
	pub fn stack_push_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_push_count_func = self.ctx.func_decl(
			self.ctx.str_sym("stack_push_count"),
			&[&instruction_sort(self.ctx).0],
			&self.ctx.int_sort(),
		);

		stack_push_count_func.apply(&[instr])
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
