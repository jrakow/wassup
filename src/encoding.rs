use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use wassup_z3::*;

struct Constants<'ctx, 'solver> {
	ctx: &'ctx Context,
	solver: &'solver Solver<'ctx>,
	word_sort: Sort<'ctx>,
	int_sort: Sort<'ctx>,
	instruction_sort: Sort<'ctx>,
	instruction_consts: Vec<FuncDecl<'ctx>>,
	stack_pop_count_func: FuncDecl<'ctx>,
	stack_push_count_func: FuncDecl<'ctx>,
	initial_stack: Vec<Ast<'ctx>>,
	stack_depth: usize,
}

impl<'ctx, 'solver> Constants<'ctx, 'solver> {
	fn new(ctx: &'ctx Context, solver: &'solver Solver<'ctx>, stack_depth: usize) -> Self {
		use Instruction::*;

		let word_sort = ctx.bv_sort(32);
		let int_sort = ctx.int_sort();
		let (instruction_sort, instruction_consts, _) = ctx.enumeration_sort(
			&ctx.string_symbol("instruction-sort"),
			&[&ctx.string_symbol("I32Const"), &ctx.string_symbol("I32Add")],
		);
		let initial_stack: Vec<_> = (0..stack_depth)
			.map(|i| ctx.fresh_const("initial-stack", word_sort))
			.collect();

		let stack_pop_count_func = ctx.func_decl(
			&ctx.string_symbol("stack-pop-count"),
			&[instruction_sort],
			int_sort,
		);
		let stack_push_count_func = ctx.func_decl(
			&ctx.string_symbol("stack-push-count"),
			&[instruction_sort],
			int_sort,
		);

		let constants = Constants {
			ctx,
			solver,
			word_sort,
			int_sort,
			instruction_sort,
			instruction_consts,
			stack_pop_count_func,
			stack_push_count_func,
			initial_stack,
			stack_depth,
		};

		let int = |i| ctx.int(i, int_sort);
		let pop_count = |i| {
			constants
				.stack_pop_count_func
				.apply(&[constants.instruction(i)])
		};
		let push_count = |i| {
			constants
				.stack_push_count_func
				.apply(&[constants.instruction(i)])
		};
		solver.assert(&pop_count(&I32Const(0)).eq(&int(0)));
		solver.assert(&push_count(&I32Const(0)).eq(&int(1)));
		solver.assert(&pop_count(&I32Add).eq(&int(2)));
		solver.assert(&push_count(&I32Add).eq(&int(1)));

		constants
	}

	fn new_state<'constants>(
		&'constants self,
		prefix: &str,
		program_length: usize,
	) -> State<'ctx, 'solver, 'constants> {
		// declare stack function
		let stack_func = self.ctx.func_decl(
			&self.ctx.string_symbol(&format!("{}stack", prefix)),
			&[
				self.int_sort, // instruction counter
				self.int_sort, // stack address
			],
			self.word_sort,
		);

		// declare stack pointer function
		let stack_pointer_func = self.ctx.func_decl(
			&self.ctx.string_symbol(&format!("{}stack-pointer", prefix)),
			&[self.int_sort],
			self.int_sort,
		);

		// declare program function
		let program_func = self.ctx.func_decl(
			&self.ctx.string_symbol(&format!("{}program", prefix)),
			&[self.int_sort],
			self.instruction_sort,
		);

		let state = State {
			constants: self,
			stack_func,
			stack_pointer_func,
			program_func,
			program_length,
		};
		state.set_initial();
		state
	}

	fn instruction(&'ctx self, i: &Instruction) -> Ast<'ctx> {
		use Instruction::*;

		match i {
			I32Const(_) => self.instruction_consts[0].apply(&[]),
			I32Add => self.instruction_consts[1].apply(&[]),
			_ => unimplemented!(),
		}
	}
}

struct State<'ctx, 'solver, 'constants> {
	constants: &'constants Constants<'ctx, 'solver>,
	stack_func: FuncDecl<'ctx>,
	stack_pointer_func: FuncDecl<'ctx>,
	program_func: FuncDecl<'ctx>,
	program_length: usize,
}

impl<'ctx, 'solver, 'constants> State<'ctx, 'solver, 'constants> {
	fn set_initial(&self) {
		// set stack(0, i) == xs[i]
		for (i, var) in self.constants.initial_stack.iter().enumerate() {
			let lhs = self.stack_func.apply(&[
				self.constants.ctx.int(0, self.constants.int_sort),
				self.constants.ctx.int(i, self.constants.int_sort),
			]);
			self.constants.solver.assert(&lhs.eq(var));
		}

		// set stack_counter(0) = 0
		self.constants.solver.assert(
			&self
				.stack_pointer_func
				.apply(&[self.constants.ctx.int(0, self.constants.int_sort)])
				.eq(&self.constants.ctx.int(0, self.constants.int_sort)),
		);
	}

	fn set_source_program(&self, program: &[Instruction]) {
		// set program_func to program
		for (index, instruction) in program.iter().enumerate() {
			let index = self.constants.ctx.int(index, self.constants.int_sort);
			let instruction = self.constants.instruction(instruction);
			self.constants
				.solver
				.assert(&self.program_func.apply(&[index]).eq(&instruction))
		}
	}
}

fn gas_particle_cost(_: &Instruction) -> usize {
	1
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn stack_pop_push_count() {
		let ctx = {
			let cfg = Config::default();
			Context::with_config(&cfg)
		};
		let solver = Solver::with_context(&ctx);
		let constants = Constants::new(&ctx, &solver, 0);

		assert!(constants.solver.check());
		let model = constants.solver.model();

		let eval_count = |func: &FuncDecl, i| {
			let ast = func.apply(&[constants.instruction(i)]);
			let ast = model.eval(&ast);
			let i: i64 = (&ast).try_into().unwrap();
			i
		};

		assert_eq!(
			eval_count(&constants.stack_pop_count_func, &Instruction::I32Add),
			2
		);
		assert_eq!(
			eval_count(&constants.stack_push_count_func, &Instruction::I32Add),
			1
		);
		assert_eq!(
			eval_count(&constants.stack_pop_count_func, &Instruction::I32Const(0)),
			0
		);
		assert_eq!(
			eval_count(&constants.stack_push_count_func, &Instruction::I32Const(0)),
			1
		);
	}

	#[test]
	fn source_program() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, 2);
		let state = constants.new_state("", program.len());
		state.set_source_program(program);

		assert!(constants.solver.check());
		let model = constants.solver.model();

		println!("{}", &solver);
	}

	#[test]
	fn initial_conditions() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, 2);
		let state = constants.new_state("", program.len());
		state.set_source_program(program);

		println!("{}", &solver);

		assert!(solver.check());
		let model = solver.model();
		println!("{}", model);
	}
}
