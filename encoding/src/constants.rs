use crate::instructions::*;
use enum_iterator::IntoEnumIterator;
use parity_wasm::elements::ValueType;
use z3::*;

pub struct Constants<'ctx> {
	pub ctx: &'ctx Context,
	pub instruction_sort: Sort<'ctx>,
	pub instruction_consts: Vec<FuncDecl<'ctx>>,
	pub instruction_testers: Vec<FuncDecl<'ctx>>,
	pub initial_stack: Vec<Ast<'ctx>>,
	pub initial_stack_types: Vec<ValueType>,
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
		let instruction_names: Vec<_> = Instruction::into_enum_iter()
			.map(|s| ctx.str_sym(&format!("{:?}", s)))
			.collect();
		let (instruction_sort, instruction_consts, instruction_testers) = ctx.enumeration_sort(
			&ctx.str_sym("Instruction"),
			&instruction_names.iter().collect::<Vec<_>>()[..],
		);
		let initial_stack: Vec<_> = (0..initial_stack_types.len())
			.map(|_| ctx.fresh_const("initial_stack", &word_sort))
			.collect();
		let params: Vec<_> = (0..n_params)
			.map(|_| ctx.fresh_const("param", &word_sort))
			.collect();

		let constants = Constants {
			ctx,
			instruction_sort,
			instruction_consts,
			instruction_testers,
			initial_stack,
			initial_stack_types: initial_stack_types.to_vec(),
			params,
		};

		for ref i in Instruction::into_enum_iter() {
			let (pops, pushs) = i.stack_pop_push_count();
			solver.assert(
				&constants
					.stack_pop_count(&constants.instruction(i))
					._eq(&ctx.from_usize(pops)),
			);
			solver.assert(
				&constants
					.stack_push_count(&constants.instruction(i))
					._eq(&ctx.from_usize(pushs)),
			);
		}

		constants
	}

	pub fn instruction(&self, i: &Instruction) -> Ast<'ctx> {
		self.instruction_consts[*i as usize].apply(&[])
	}

	pub fn stack_pop_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_pop_count_func = self.ctx.func_decl(
			self.ctx.str_sym("stack_pop_count"),
			&[&self.instruction_sort],
			&self.ctx.int_sort(),
		);

		stack_pop_count_func.apply(&[instr])
	}

	pub fn stack_push_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_push_count_func = self.ctx.func_decl(
			self.ctx.str_sym("stack_push_count"),
			&[&self.instruction_sort],
			&self.ctx.int_sort(),
		);

		stack_push_count_func.apply(&[instr])
	}

	pub fn set_params(&self, solver: &Solver, params: &[u32]) {
		for (i, v) in params.iter().enumerate() {
			let v = self.ctx.from_u32(*v).int2bv(32);

			solver.assert(&self.params[i]._eq(&v));
		}
	}
}
