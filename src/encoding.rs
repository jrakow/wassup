use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use wassup_z3::*;

fn instruction_to_index(i: &Instruction) -> usize {
	use Instruction::*;

	match i {
		I32Const(_) => 0,
		I32Add => 1,
		Nop => 2,
		_ => unimplemented!(),
	}
}

fn stack_pop_push_count(i: &Instruction) -> (isize, isize) {
	use Instruction::*;

	match i {
		I32Const(_) => (0, 1),
		I32Add => (2, 1),
		Nop => (0, 0),
		_ => unimplemented!(),
	}
}

fn iter_intructions() -> impl Iterator<Item = &'static Instruction> {
	use Instruction::*;

	static INSTRUCTIONS: &[Instruction] = &[I32Const(0), I32Add, Nop];
	INSTRUCTIONS.iter()
}

fn stack_depth(program: &[Instruction]) -> usize {
	let mut stack_pointer = 0;
	let mut lowest = 0;
	for i in program {
		let (pops, pushs) = stack_pop_push_count(i);
		lowest = std::cmp::min(lowest, stack_pointer - pops);
		stack_pointer = stack_pointer - pops + pushs;
	}
	lowest.abs().try_into().unwrap()
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
	in_range_func: FuncDecl<'ctx>,
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
			&[
				ctx.string_symbol("I32Const"),
				ctx.string_symbol("I32Add"),
				ctx.string_symbol("Nop"),
			],
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
		let in_range_func = ctx.func_decl(
			ctx.string_symbol("in-range"),
			&[int_sort, int_sort, int_sort],
			ctx.bool_sort(),
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
			in_range_func,
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

		// define in-range(a, b, c): a <= b && b < c
		let c = ctx.bound(0, int_sort);
		let b = ctx.bound(1, int_sort);
		let a = ctx.bound(2, int_sort);
		let body = constants
			.in_range(a.clone(), b.clone(), c.clone())
			.eq(ctx.and(&[ctx.le(a, b.clone()), ctx.lt(b, c)]));
		solver.assert(ctx.forall(
			&[int_sort, int_sort, int_sort],
			&[
				ctx.string_symbol("a"),
				ctx.string_symbol("b"),
				ctx.string_symbol("c"),
			],
			body,
		));

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

	fn in_range(&self, a: Ast, b: Ast, c: Ast) -> Ast {
		self.in_range_func.apply(&[a, b, c])
	}

	fn int(&self, i: isize) -> Ast {
		self.ctx.int(i, self.ctx.int_sort())
	}

	fn uint(&self, i: usize) -> Ast {
		self.ctx.uint(i, self.ctx.int_sort())
	}

	fn int2word(&self, i: Ast) -> Ast {
		self.ctx.int2bv(32, i)
	}
}

struct State<'ctx, 'solver, 'constants> {
	ctx: &'ctx Context,
	solver: &'solver Solver<'ctx>,
	constants: &'constants Constants<'ctx, 'solver>,

	prefix: String,

	stack_func: FuncDecl<'ctx>,
	// stack_pointer - 1 is top of stack
	stack_pointer_func: FuncDecl<'ctx>,

	program_length: usize,
	program_func: FuncDecl<'ctx>,
	push_constants_func: FuncDecl<'ctx>,

	transition_func: FuncDecl<'ctx>,
	transition_stack_pointer_func: FuncDecl<'ctx>,
	transition_stack_func: FuncDecl<'ctx>,
	preserve_stack_func: FuncDecl<'ctx>,
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
		// declare push_constants function
		let push_constants_func = ctx.func_decl(
			ctx.string_symbol(&(prefix.to_owned() + "push-constants")),
			&[constants.int_sort],
			constants.word_sort,
		);

		// declare transition functions
		let transition_func = ctx.func_decl(
			ctx.string_symbol(&(prefix.to_owned() + "transition")),
			&[constants.int_sort],
			constants.ctx.bool_sort(),
		);
		let transition_stack_pointer_func = ctx.func_decl(
			ctx.string_symbol(&(prefix.to_owned() + "transition-stack-pointer")),
			&[constants.int_sort],
			ctx.bool_sort(),
		);
		let transition_stack_func = ctx.func_decl(
			ctx.string_symbol(&(prefix.to_owned() + "transition-stack")),
			&[constants.int_sort],
			ctx.bool_sort(),
		);
		let preserve_stack_func = ctx.func_decl(
			ctx.string_symbol(&(prefix.to_owned() + "preserve-stack")),
			&[constants.int_sort],
			ctx.bool_sort(),
		);

		let state = State {
			ctx,
			solver,
			constants,

			prefix: prefix.to_string(),
			stack_func,
			stack_pointer_func,

			program_length,
			program_func,
			push_constants_func,

			transition_func,
			transition_stack_pointer_func,
			transition_stack_func,
			preserve_stack_func,
		};

		state.set_initial();

		state.define_transition_stack_pointer();
		state.define_transition_stack();
		state.define_transition();
		state.define_preserve_stack();

		state
	}

	fn set_initial(&self) {
		// set stack(0, i) == xs[i]
		for (i, var) in self.constants.initial_stack.iter().enumerate() {
			self.solver.assert(
				self.stack(self.constants.uint(0), self.constants.uint(i))
					.eq(var.clone()),
			);
		}

		// set stack_counter(0) = 0
		self.solver.assert(
			self.stack_pointer(self.constants.uint(0))
				.eq(self.constants.uint(self.constants.stack_depth)),
		);
	}

	fn set_source_program(&self, program: &[Instruction]) {
		assert_eq!(self.program_length, program.len());

		// set program_func to program
		for (pc, instruction) in program.iter().enumerate() {
			let instruction = self.constants.instruction(instruction);
			self.solver
				.assert(self.program(self.constants.uint(pc)).eq(instruction))
		}

		// set push_constants function
		for (pc, instr) in program.iter().enumerate() {
			if let Instruction::I32Const(i) = instr {
				let i = self
					.constants
					.int2word(self.constants.int((*i).try_into().unwrap()));
				self.solver
					.assert(self.push_constants(self.constants.uint(pc)).eq(i));
			}
		}
	}

	fn define_transition_stack_pointer(&self) {
		for pc in 0..self.program_length {
			// encode stack_pointer change
			let stack_pointer = self.stack_pointer(self.constants.uint(pc));
			let stack_pointer_next = self.stack_pointer(self.constants.uint(pc + 1));

			let instruction = self.program(self.constants.uint(pc));
			let pop_count = self.constants.stack_pop_count(instruction.clone());
			let push_count = self.constants.stack_push_count(instruction);

			let new_pointer = stack_pointer + push_count - pop_count;

			self.solver.assert(self.ctx.iff(
				self.transition_stack_pointer(self.constants.uint(pc)),
				stack_pointer_next.eq(new_pointer),
			));
		}
	}

	fn define_transition_stack(&self) {
		for pc in 0..self.program_length {
			// constants
			let instr = self.program(self.constants.uint(pc));
			let stack_pointer = self.stack_pointer(self.constants.uint(pc));

			// encode instruction effect
			let new_stack_pointer = self.stack_pointer(self.constants.uint(pc + 1));

			// instr == Nop

			// instr == Add implies stack(pc + 1, new_stack_pointer - 1) == stack(pc, stack_pointer - 1) + stack(pc, stack_pointer - 2)
			let lhs = instr.eq(self.constants.instruction(&Instruction::I32Add));

			let sum = self.ctx.bvadd(
				self.stack(
					self.constants.uint(pc),
					stack_pointer.clone() - self.constants.int(1),
				),
				self.stack(
					self.constants.uint(pc),
					stack_pointer.clone() - self.constants.int(2),
				),
			);
			let rhs = self
				.stack(
					self.constants.uint(pc + 1),
					new_stack_pointer.clone() - self.constants.int(1),
				)
				.eq(sum);
			let add_effect = self.ctx.implies(lhs, rhs);

			// instr == Const implies stack(pc + 1, new_stack_pointer - 1) == consts(pc)
			let lhs = instr.eq(self.constants.instruction(&Instruction::I32Const(0)));
			let rhs = self
				.stack(
					self.constants.uint(pc + 1),
					new_stack_pointer - self.constants.int(1),
				)
				.eq(self.push_constants(self.constants.uint(pc)));
			let const_effect = self.ctx.implies(lhs, rhs);

			let instruction_effect = self.ctx.and(&[add_effect, const_effect]);

			self.solver.assert(self.ctx.iff(
				self.transition_stack(self.constants.uint(pc)),
				self.ctx.and(&[
					self.preserve_stack(self.constants.uint(pc)),
					instruction_effect,
				]),
			));
		}
	}

	fn define_transition(&self) {
		for pc in 0..self.program_length {
			self.solver.assert(self.ctx.iff(
				self.transition(self.constants.uint(pc)),
				self.ctx.and(&[
					self.transition_stack_pointer(self.constants.uint(pc)),
					self.transition_stack(self.constants.uint(pc)),
				]),
			));
		}
	}

	fn define_preserve_stack(&self) {
		for pc in 0..self.program_length {
			// constants
			let instr = self.program(self.constants.uint(pc));
			let stack_pointer = self.stack_pointer(self.constants.uint(pc));

			// preserve stack values stack(_, 0)..=stack(_, stack_pointer - pops - 1)
			let n = self.ctx.bound(0, self.constants.int_sort);

			let n_in_range = self.constants.in_range(
				self.constants.uint(0),
				n.clone(),
				stack_pointer.clone() - self.constants.stack_pop_count(instr.clone()),
			);
			let slot_preserved = self
				.stack(self.constants.uint(pc), n.clone())
				.eq(self.stack(self.constants.uint(pc + 1), n));
			let body = self.ctx.implies(n_in_range, slot_preserved);

			// forall n
			let stack_is_preserved = self.ctx.forall(
				&[self.constants.int_sort],
				&[self.ctx.string_symbol("n")],
				body,
			);

			self.solver.assert(self.ctx.iff(
				self.preserve_stack(self.constants.uint(pc)),
				stack_is_preserved,
			));
		}
	}

	fn transition(&self, pc: Ast) -> Ast {
		self.transition_func.apply(&[pc])
	}

	fn transition_stack_pointer(&self, pc: Ast) -> Ast {
		self.transition_stack_pointer_func.apply(&[pc])
	}

	fn transition_stack(&self, pc: Ast) -> Ast {
		self.transition_stack_func.apply(&[pc])
	}

	fn preserve_stack(&self, pc: Ast) -> Ast {
		self.preserve_stack_func.apply(&[pc])
	}

	fn stack_pointer(&self, pc: Ast) -> Ast {
		self.stack_pointer_func.apply(&[pc])
	}

	fn stack(&self, pc: Ast, index: Ast) -> Ast {
		self.stack_func.apply(&[pc, index])
	}

	fn program(&self, pc: Ast) -> Ast {
		self.program_func.apply(&[pc])
	}

	fn push_constants(&self, pc: Ast) -> Ast {
		self.push_constants_func.apply(&[pc])
	}
}

fn define_equivalent<'ctx>(lhs: &'ctx State, rhs: &'ctx State) -> FuncDecl<'ctx> {
	let ctx = lhs.ctx;
	let constants = rhs.constants;
	let solver = lhs.solver;

	let lhs_pc = ctx.r#const(ctx.string_symbol("lhs_pc"), constants.int_sort);
	let rhs_pc = ctx.r#const(ctx.string_symbol("rhs_pc"), constants.int_sort);

	let lhs_program_length = constants.uint(lhs.program_length);
	let lhs_pc_in_range = constants.in_range(constants.uint(0), lhs_pc.clone(), lhs_program_length);
	let rhs_program_length = constants.uint(rhs.program_length);
	let rhs_pc_in_range = constants.in_range(constants.uint(0), rhs_pc.clone(), rhs_program_length);

	let name = &(lhs.prefix.clone() + &rhs.prefix + "equivalent");
	let equivalent_func = ctx.func_decl(
		ctx.string_symbol(name),
		&[constants.int_sort, constants.int_sort],
		ctx.bool_sort(),
	);

	let stack_pointers_equal = lhs
		.stack_pointer_func
		.apply(&[lhs_pc.clone()])
		.eq(rhs.stack_pointer_func.apply(&[rhs_pc.clone()]));

	let stacks_equal = {
		// for 0 <= n < stack_pointer
		let n = ctx.r#const(ctx.string_symbol("n"), constants.int_sort);
		let n_in_range = constants.in_range(
			constants.uint(0),
			n.clone(),
			lhs.stack_pointer_func.apply(&[lhs_pc.clone()]),
		);

		// lhs-stack(lhs_pc, n) ==  rhs-stack(rhs_pc, n)
		let condition = lhs
			.stack_func
			.apply(&[lhs_pc.clone(), n.clone()])
			.eq(rhs.stack_func.apply(&[rhs_pc.clone(), n.clone()]));

		ctx.forall_const(&[n], ctx.implies(n_in_range, condition))
	};

	//	let bounds = ctx.and(&[lhs_pc_in_range, rhs_pc_in_range]);
	let expected = equivalent_func.apply(&[lhs_pc.clone(), rhs_pc.clone()]);
	let actual = ctx.and(&[stack_pointers_equal, stacks_equal]);
	solver.assert(ctx.forall_const(&[lhs_pc, rhs_pc], expected.eq(actual)));

	equivalent_func
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
	fn stack_depth_test() {
		let program = &[];
		assert_eq!(stack_depth(program), 0);

		let program = &[Instruction::I32Add];
		assert_eq!(stack_depth(program), 2);

		let program = &[Instruction::I32Const(1), Instruction::I32Add];
		assert_eq!(stack_depth(program), 1);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(1),
			Instruction::I32Const(1),
			Instruction::I32Add,
		];
		assert_eq!(stack_depth(program), 0);
	}

	#[test]
	fn source_program() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::Nop,
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		assert!(constants.solver.check());
		let model = constants.solver.model();

		for (i, instr) in program.iter().enumerate() {
			let instr_enc = state.program(constants.uint(i));
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.model();

		let stack_pointer = state.stack_pointer(constants.uint(0));
		let stack_pointer_int: i64 = model.eval(stack_pointer).try_into().unwrap();
		assert_eq!(stack_pointer_int, 0);
	}

	#[test]
	fn stack_pointer_transition() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		for i in 0..program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				state.transition_stack_pointer(constants.uint(i)),
			));
		}

		assert!(solver.check());
		let model = solver.model();

		let eval = |ast| -> i64 {
			let evaled = model.eval(ast);
			evaled.try_into().unwrap()
		};
		assert_eq!(eval(state.stack_pointer(constants.uint(0))), 0);
		assert_eq!(eval(state.stack_pointer(constants.uint(1))), 1);
		assert_eq!(eval(state.stack_pointer(constants.uint(2))), 2);
		assert_eq!(eval(state.stack_pointer(constants.uint(3))), 1);
	}

	#[test]
	fn consts() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[Instruction::I32Const(1), Instruction::I32Const(2)];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.model();

		let eval = |ast| -> i64 {
			let evaled = model.eval(ast);
			evaled.try_into().unwrap()
		};
		assert_eq!(eval(state.push_constants(constants.uint(1))), 2);
		assert_eq!(eval(state.push_constants(constants.uint(0))), 1);
	}

	#[test]
	fn transition_const() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		for i in 0..program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				state.transition(constants.uint(i)),
			));
		}

		assert!(solver.check());
		let model = solver.model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast);
			evaled.try_into().unwrap()
		};
		assert_eq!(eval_int(state.stack_pointer(constants.uint(1))), 1);

		let eval_bv = |ast| -> i64 {
			let evaled = model.eval(ctx.bv2int(ast));
			evaled.try_into().unwrap()
		};
		assert_eq!(eval_bv(state.stack(constants.uint(1), constants.int(0))), 1);
	}

	#[test]
	fn transition_add_consts() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::Nop,
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		for i in 0..program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				state.transition(constants.uint(i)),
			));
		}

		assert!(solver.check());
		let model = solver.model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast);
			evaled.try_into().unwrap()
		};

		let eval_bv = |ast| -> i64 {
			let evaled = model.eval(ctx.bv2int(ast));
			evaled.try_into().unwrap()
		};
		assert_eq!(eval_bv(state.stack(constants.uint(1), constants.int(0))), 1);
		assert_eq!(eval_bv(state.stack(constants.uint(2), constants.int(0))), 1);
		assert_eq!(eval_bv(state.stack(constants.uint(3), constants.int(0))), 1);
		assert_eq!(eval_bv(state.stack(constants.uint(3), constants.int(1))), 2);
		assert_eq!(eval_bv(state.stack(constants.uint(4), constants.int(0))), 3);
	}

	#[test]
	fn transition_add() {
		let ctx = Context::with_config(&Config::default());
		let solver = Solver::with_context(&ctx);

		let program = &[Instruction::I32Add];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let state = State::new(&ctx, &solver, &constants, "", program.len());

		state.set_source_program(program);

		for i in 0..program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				state.transition(constants.uint(i)),
			));
		}

		assert!(solver.check());
		let model = solver.model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast);
			evaled.try_into().unwrap()
		};

		let eval_bv = |ast| -> i64 {
			let evaled = model.eval(ctx.bv2int(ast));
			evaled.try_into().unwrap()
		};
		let sum = ctx.bvadd(
			constants.initial_stack[0].clone(),
			constants.initial_stack[1].clone(),
		);
		assert_eq!(
			eval_bv(state.stack(constants.uint(1), constants.int(0))),
			eval_bv(sum)
		);
	}

	#[test]
	fn equivalent_reflexive() {
		let config = Config::default();
		let ctx = Context::with_config(&config);
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::Nop,
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program));
		let source_state = State::new(&ctx, &solver, &constants, "source-", program.len());
		let target_state = State::new(&ctx, &solver, &constants, "target-", program.len());

		source_state.set_source_program(program);
		target_state.set_source_program(program);

		for i in 0..program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				source_state.transition(constants.uint(i)),
			));
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				target_state.transition(constants.uint(i)),
			));
		}

		let equivalent_func = define_equivalent(&source_state, &target_state);

		assert!(solver.check());
		let model = solver.model();

		let equivalent = |i, j| equivalent_func.apply(&[constants.uint(i), constants.uint(j)]);

		for i in 0..program.len() {
			let s = equivalent(i, i);
			let evaled = model.eval(s);
			let equiv: bool = evaled.try_into().unwrap();
			assert!(equiv);
		}
	}

	#[test]
	fn equivalent_nop_no_effect() {
		let config = Config::default();
		let ctx = Context::with_config(&config);
		let solver = Solver::with_context(&ctx);

		let lhs_program = &[Instruction::I32Const(1), Instruction::Nop];
		let rhs_program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(lhs_program));
		let lhs_state = State::new(&ctx, &solver, &constants, "source-", lhs_program.len());
		let rhs_state = State::new(&ctx, &solver, &constants, "target-", rhs_program.len());
		lhs_state.set_source_program(lhs_program);
		rhs_state.set_source_program(rhs_program);

		for i in 0..lhs_program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				lhs_state.transition(constants.uint(i)),
			));
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				rhs_state.transition(constants.uint(i)),
			));
		}

		let equivalent_func = define_equivalent(&lhs_state, &rhs_state);

		assert!(solver.check());
		let model = solver.model();

		let s = equivalent_func.apply(&[constants.uint(2), constants.uint(1)]);
		let evaled = model.eval(s);
		let equiv: bool = evaled.try_into().unwrap();
		assert!(equiv);
	}

	#[test]
	fn equivalent_stack_unequal() {
		let config = Config::default();
		let ctx = Context::with_config(&config);
		let solver = Solver::with_context(&ctx);

		let lhs_program = &[Instruction::I32Const(1), Instruction::I32Const(2)];
		let rhs_program = &[Instruction::I32Const(2), Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(lhs_program));
		let lhs_state = State::new(&ctx, &solver, &constants, "source-", lhs_program.len());
		let rhs_state = State::new(&ctx, &solver, &constants, "target-", rhs_program.len());
		lhs_state.set_source_program(lhs_program);
		rhs_state.set_source_program(rhs_program);

		for i in 0..lhs_program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				lhs_state.transition(constants.uint(i)),
			));
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				rhs_state.transition(constants.uint(i)),
			));
		}

		let equivalent_func = define_equivalent(&lhs_state, &rhs_state);

		assert!(solver.check());
		let model = solver.model();

		let s = equivalent_func.apply(&[constants.uint(2), constants.uint(2)]);
		let evaled = model.eval(s);
		let equiv: bool = evaled.try_into().unwrap();
		assert!(!equiv);
	}

	#[test]
	fn equivalent_stack_pointer_unequal() {
		let config = Config::default();
		let ctx = Context::with_config(&config);
		let solver = Solver::with_context(&ctx);

		let lhs_program = &[Instruction::I32Const(1), Instruction::I32Const(2)];
		let rhs_program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(lhs_program));
		let lhs_state = State::new(&ctx, &solver, &constants, "source-", lhs_program.len());
		let rhs_state = State::new(&ctx, &solver, &constants, "target-", rhs_program.len());
		lhs_state.set_source_program(lhs_program);
		rhs_state.set_source_program(rhs_program);

		for i in 0..lhs_program.len() {
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				lhs_state.transition(constants.uint(i)),
			));
			solver.assert(ctx.forall_const(
				&constants.initial_stack[..],
				rhs_state.transition(constants.uint(i)),
			));
		}

		let equivalent_func = define_equivalent(&lhs_state, &rhs_state);

		assert!(solver.check());
		let model = solver.model();

		let s = equivalent_func.apply(&[constants.uint(2), constants.uint(1)]);
		let evaled = model.eval(s);
		let equiv: bool = evaled.try_into().unwrap();
		assert!(!equiv);
	}
}
