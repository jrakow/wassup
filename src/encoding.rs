use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use wassup_z3::*;

fn instruction_to_index(i: &Instruction) -> usize {
	use Instruction::*;

	match i {
		I32Const(_) => 0,
		I32Add => 1,
		_ => unimplemented!(),
	}
}

fn stack_pop_push_count(i: &Instruction) -> (usize, usize) {
	use Instruction::*;

	match i {
		I32Const(_) => (0, 1),
		I32Add => (2, 1),
		_ => unimplemented!(),
	}
}

fn iter_intructions() -> impl Iterator<Item = &'static Instruction> {
	use Instruction::*;

	static INSTRUCTIONS: &[Instruction] = &[I32Const(0), I32Add];
	INSTRUCTIONS.iter()
}

struct Constants<'ctx, 'solver> {
	ctx: &'ctx Context,
	solver: &'solver Solver<'ctx>,

	word_sort: Sort<'ctx>,
	int_sort: Sort<'ctx>,
	instruction_sort: Sort<'ctx>,
	instruction_consts: Vec<FuncDecl<'ctx>>,
	instruction_testers: Vec<FuncDecl<'ctx>>,
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
		let (instruction_sort, instruction_consts, instruction_testers) = ctx.enumeration_sort(
			ctx.string_symbol("instruction-sort"),
			&[ctx.string_symbol("I32Const"), ctx.string_symbol("I32Add")],
		);
		let initial_stack: Vec<_> = (0..stack_depth)
			.map(|i| ctx.fresh_const("initial-stack", word_sort))
			.collect();

		let stack_pop_count_func = ctx.func_decl(
			ctx.string_symbol("stack-pop-count"),
			&[instruction_sort],
			int_sort,
		);
		let stack_push_count_func = ctx.func_decl(
			ctx.string_symbol("stack-push-count"),
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
			instruction_testers,
			stack_pop_count_func,
			stack_push_count_func,
			initial_stack,
			stack_depth,
		};

		let int = |i| ctx.int(i, int_sort);

		for i in iter_intructions() {
			let (pops, pushs) = stack_pop_push_count(i);
			solver.assert(
				constants
					.stack_pop_count(constants.instruction(i))
					.eq(int(pops)),
			);
			solver.assert(
				constants
					.stack_push_count(constants.instruction(i))
					.eq(int(pushs)),
			);
		}

		constants
	}

	fn instruction(&'ctx self, i: &Instruction) -> Ast {
		self.instruction_consts[instruction_to_index(i)].apply(&[])
	}

	fn stack_pop_count(&self, instr: Ast) -> Ast {
		self.stack_pop_count_func.apply(&[instr])
	}

	fn stack_push_count(&self, instr: Ast) -> Ast {
		self.stack_push_count_func.apply(&[instr])
	}

	fn int(&self, i: usize) -> Ast {
		self.ctx.int(i, self.ctx.int_sort())
	}
}

struct State<'ctx, 'solver, 'constants> {
	ctx: &'ctx Context,
	solver: &'solver Solver<'ctx>,
	constants: &'constants Constants<'ctx, 'solver>,

	prefix: String,
	stack_func: FuncDecl<'ctx>,
	stack_pointer_func: FuncDecl<'ctx>,
	program_func: FuncDecl<'ctx>,
	program_length: usize,
}

impl<'ctx, 'solver, 'constants> State<'ctx, 'solver, 'constants> {
	fn new(
		ctx: &'ctx Context,
		solver: &'solver Solver<'ctx>,
		constants: &'constants Constants<'ctx, 'solver>,
		prefix: &str,
		program_length: usize,
	) -> Self {
		// declare stack function
		let stack_func = ctx.func_decl(
			ctx.string_symbol(&format!("{}stack", prefix)),
			&[
				constants.int_sort, // instruction counter
				constants.int_sort, // stack address
			],
			constants.word_sort,
		);

		// declare stack pointer function
		let stack_pointer_func = ctx.func_decl(
			ctx.string_symbol(&format!("{}stack-pointer", prefix)),
			&[constants.int_sort],
			constants.int_sort,
		);

		// declare program function
		let program_func = ctx.func_decl(
			ctx.string_symbol(&format!("{}program", prefix)),
			&[constants.int_sort],
			constants.instruction_sort,
		);

		let state = State {
			ctx,
			solver,
			constants,

			prefix: prefix.to_string(),
			stack_func,
			stack_pointer_func,
			program_func,
			program_length,
		};
		state.set_initial();
		state
	}

	fn set_initial(&self) {
		// set stack(0, i) == xs[i]
		for (i, var) in self.constants.initial_stack.iter().enumerate() {
			self.solver.assert(self.stack(0, i).eq(var.clone()));
		}

		// set stack_counter(0) = 0
		self.solver
			.assert(self.stack_pointer(0).eq(self.constants.int(0)));
	}

	fn set_source_program(&self, program: &[Instruction]) {
		// set program_func to program
		for (index, instruction) in program.iter().enumerate() {
			let instruction = self.constants.instruction(instruction);
			self.solver.assert(self.program(index).eq(instruction))
		}
	}

	fn stack_pointer_transition_condition(&self) -> Ast {
		let mut conditions = vec![];

		for i in 0..self.program_length {
			// encode stack_pointer change
			let stack_pointer = self.stack_pointer(i);
			let stack_pointer_next = self.stack_pointer(i + 1);

			let instruction = self.program(i);
			let pop_count = self.constants.stack_pop_count(instruction.clone());
			let push_count = self.constants.stack_push_count(instruction);

			let new_pointer = stack_pointer + push_count - pop_count;
			conditions.push(stack_pointer_next.eq(new_pointer));
		}

		self.ctx.and(&conditions[..])
	}

	fn stack_pointer(&self, index: usize) -> Ast {
		self.stack_pointer_func.apply(&[self.constants.int(index)])
	}

	fn stack(&self, pc: usize, index: usize) -> Ast {
		self.stack_func
			.apply(&[self.constants.int(pc), self.constants.int(index)])
	}

	fn program(&self, pc: usize) -> Ast {
		self.program_func.apply(&[self.constants.int(pc)])
	}
}

fn gas_particle_cost(_: &Instruction) -> usize {
	1
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_stack_pop_push_count() {
		let ctx = {
			let cfg = Config::default();
			Context::with_config(&cfg)
		};
		let solver = Solver::with_context(&ctx);
		let constants = Constants::new(&ctx, &solver, 0);

		assert!(constants.solver.check());
		let model = constants.solver.model();

		let eval = |ast: &Ast| -> i64 {
			let ast = model.eval(ast.clone());
			ast.try_into().unwrap()
		};

		for i in iter_intructions() {
			let (pops, pushs) = stack_pop_push_count(i);
			let (pops, pushs) = (pops as i64, pushs as i64);
			assert_eq!(
				eval(&constants.stack_pop_count(constants.instruction(i))),
				pops
			);
			assert_eq!(
				eval(&constants.stack_push_count(constants.instruction(i))),
				pushs
			);
		}

		assert_eq!(
			eval(&constants.stack_pop_count(constants.instruction(&Instruction::I32Add))),
			2
		);
		assert_eq!(
			eval(&constants.stack_push_count(constants.instruction(&Instruction::I32Add))),
			1
		);
		assert_eq!(
			eval(&constants.stack_pop_count(constants.instruction(&Instruction::I32Const(0)))),
			0
		);
		assert_eq!(
			eval(&constants.stack_push_count(constants.instruction(&Instruction::I32Const(0)))),
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
		let state = State::new(&ctx, &solver, &constants, "", program.len());
		state.set_source_program(program);

		assert!(constants.solver.check());
		let model = constants.solver.model();

		for (i, instr) in program.iter().enumerate() {
			let instr_enc = state.program(i);
			let is_equal = constants.instruction_testers[instruction_to_index(instr)]
				.apply(&[instr_enc.clone()]);
			let b: bool = model.eval(is_equal).try_into().unwrap();
			assert!(b);
		}
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
		let state = State::new(&ctx, &solver, &constants, "", program.len());
		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.model();

		let stack_pointer = state.stack_pointer(0);
		let stack_pointer_int: i64 = model.eval(stack_pointer).try_into().unwrap();
		assert_eq!(stack_pointer_int, 0);
	}

	#[test]
	fn stack_pointer_transition() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, 2);
		let state = State::new(&ctx, &solver, &constants, "", program.len());
		state.set_source_program(program);
		solver.assert(state.stack_pointer_transition_condition());

		assert!(solver.check());
		let model = solver.model();

		let eval = |ast| -> i64 {
			let evaled = model.eval(ast);
			evaled.try_into().unwrap()
		};
		assert_eq!(eval(state.stack_pointer(0)), 0);
		assert_eq!(eval(state.stack_pointer(1)), 1);
	}
}
