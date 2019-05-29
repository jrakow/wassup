use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use z3::*;

fn instruction_to_index(i: &Instruction) -> usize {
	use Instruction::*;

	match i {
		I32Const(_) => 0,
		I32Add => 1,
		Nop => 2,
		_ => unimplemented!(),
	}
}

fn stack_pop_push_count(i: &Instruction) -> (u64, u64) {
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

fn stack_depth(program: &[Instruction]) -> u64 {
	let mut stack_pointer: isize = 0;
	let mut lowest: isize = 0;
	for i in program {
		let (pops, pushs) = stack_pop_push_count(i);
		let (pops, pushs) = (pops as isize, pushs as isize);
		lowest = std::cmp::min(lowest, stack_pointer - pops);
		stack_pointer = stack_pointer - pops + pushs;
	}
	lowest.abs().try_into().unwrap()
}

struct Constants<'ctx, 'solver> {
	ctx: &'ctx Context,
	solver: &'solver Solver<'ctx>,

	word_sort: Sort<'ctx>,
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

		let word_sort = ctx.bitvector_sort(32);
		let (instruction_sort, instruction_consts, instruction_testers) = ctx.enumeration_sort(
			&ctx.str_sym("instruction-sort"),
			&[
				&ctx.str_sym("I32Const"),
				&ctx.str_sym("I32Add"),
				&ctx.str_sym("Nop"),
			],
		);
		let initial_stack: Vec<_> = (0..stack_depth)
			.map(|i| ctx.fresh_const("initial-stack", &word_sort))
			.collect();

		let stack_pop_count_func = ctx.func_decl(
			ctx.str_sym("stack-pop-count"),
			&[&instruction_sort],
			&ctx.int_sort(),
		);
		let stack_push_count_func = ctx.func_decl(
			ctx.str_sym("stack-push-count"),
			&[&instruction_sort],
			&ctx.int_sort(),
		);
		let in_range_func = ctx.func_decl(
			ctx.str_sym("in-range"),
			&[&ctx.int_sort(), &ctx.int_sort(), &ctx.int_sort()],
			&ctx.bool_sort(),
		);

		let constants = Constants {
			ctx,
			solver,
			word_sort,
			instruction_sort,
			instruction_consts,
			instruction_testers,
			stack_pop_count_func,
			stack_push_count_func,
			in_range_func,
			initial_stack,
			stack_depth,
		};

		for i in iter_intructions() {
			let (pops, pushs) = stack_pop_push_count(i);
			solver.assert(
				&constants
					.stack_pop_count(&constants.instruction(i))
					._eq(&ctx.from_u64(pops)),
			);
			solver.assert(
				&constants
					.stack_push_count(&constants.instruction(i))
					._eq(&ctx.from_u64(pushs)),
			);
		}

		// define in-range(a, b, c): a <= b && b < c
		let a = ctx.named_int_const("a");
		let b = ctx.named_int_const("b");
		let c = ctx.named_int_const("c");
		let body = constants
			.in_range(&a, &b, &c)
			._eq(&a.le(&b).and(&[&b.lt(&c)]));
		solver.assert(&ctx.forall_const(&[&a, &b, &c], &body));

		constants
	}

	fn instruction(&'ctx self, i: &Instruction) -> Ast {
		self.instruction_consts[instruction_to_index(i)].apply(&[])
	}

	fn stack_pop_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		self.stack_pop_count_func.apply(&[instr])
	}

	fn stack_push_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		self.stack_push_count_func.apply(&[instr])
	}

	fn in_range(&self, a: &Ast<'ctx>, b: &Ast<'ctx>, c: &Ast<'ctx>) -> Ast<'ctx> {
		self.in_range_func.apply(&[a, b, c])
	}

	fn int2word(&self, i: &Ast<'ctx>) -> Ast<'ctx> {
		i.int2bv(32)
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

	program_length: Ast<'ctx>,
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
	) -> Self {
		// declare stack function
		let stack_func = ctx.func_decl(
			ctx.str_sym(&format!("{}stack", prefix)),
			&[
				&ctx.int_sort(), // instruction counter
				&ctx.int_sort(), // stack address
			],
			&constants.word_sort,
		);

		// declare stack pointer function
		let stack_pointer_func = ctx.func_decl(
			ctx.str_sym(&format!("{}stack-pointer", prefix)),
			&[&ctx.int_sort()],
			&ctx.int_sort(),
		);

		// declare program function
		let program_func = ctx.func_decl(
			ctx.str_sym(&format!("{}program", prefix)),
			&[&ctx.int_sort()],
			&constants.instruction_sort,
		);
		// declare push_constants function
		let push_constants_func = ctx.func_decl(
			ctx.str_sym(&(prefix.to_owned() + "push-constants")),
			&[&ctx.int_sort()],
			&constants.word_sort,
		);
		// declare program length constant
		let program_length = ctx.named_int_const(&(prefix.to_owned() + "program-length"));

		// declare transition functions
		let transition_func = ctx.func_decl(
			ctx.str_sym(&(prefix.to_owned() + "transition")),
			&[&ctx.int_sort()],
			&ctx.bool_sort(),
		);
		let transition_stack_pointer_func = ctx.func_decl(
			ctx.str_sym(&(prefix.to_owned() + "transition-stack-pointer")),
			&[&ctx.int_sort()],
			&ctx.bool_sort(),
		);
		let transition_stack_func = ctx.func_decl(
			ctx.str_sym(&(prefix.to_owned() + "transition-stack")),
			&[&ctx.int_sort(), &constants.instruction_sort],
			&ctx.bool_sort(),
		);
		let preserve_stack_func = ctx.func_decl(
			ctx.str_sym(&(prefix.to_owned() + "preserve-stack")),
			&[&ctx.int_sort()],
			&ctx.bool_sort(),
		);

		let state = State {
			ctx,
			solver,
			constants,

			prefix: prefix.to_string(),
			stack_func,
			stack_pointer_func,

			program_func,
			push_constants_func,
			program_length,

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
				&self
					.stack(&self.ctx.from_u64(0), &self.ctx.from_u64(i as _))
					._eq(&var),
			);
		}

		// set stack_counter(0) = 0
		self.solver.assert(
			&self
				.stack_pointer(&self.ctx.from_u64(0))
				._eq(&self.ctx.from_u64(self.constants.stack_depth as _)),
		);
	}

	fn set_source_program(&self, program: &[Instruction]) {
		// set program_func to program
		for (pc, instruction) in program.iter().enumerate() {
			let instruction = self.constants.instruction(instruction);
			self.solver
				.assert(&self.program(&self.ctx.from_u64(pc as _))._eq(&instruction))
		}

		// set push_constants function
		for (pc, instr) in program.iter().enumerate() {
			if let Instruction::I32Const(i) = instr {
				let i = self
					.constants
					.int2word(&self.ctx.from_i64((*i).try_into().unwrap()));
				self.solver
					.assert(&self.push_constants(&self.ctx.from_u64(pc as _))._eq(&i));
			}
		}

		// set length
		self.solver.assert(
			&self
				.program_length
				._eq(&self.ctx.from_u64(program.len() as u64)),
		)
	}

	fn define_transition_stack_pointer(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = self
			.constants
			.in_range(&self.ctx.from_u64(0), &pc, &self.program_length);

		// encode stack_pointer change
		let stack_pointer = self.stack_pointer(&pc);
		let stack_pointer_next = self.stack_pointer(&pc.add(&[&self.ctx.from_u64(1)]));

		let instruction = self.program(&pc);
		let pop_count = self.constants.stack_pop_count(&instruction);
		let push_count = self.constants.stack_push_count(&instruction);

		let new_pointer = stack_pointer.add(&[&push_count]).sub(&[&pop_count]);

		let definition = self
			.transition_stack_pointer(&pc)
			.iff(&stack_pointer_next._eq(&new_pointer));

		self.solver.assert(
			&self
				.ctx
				.forall_const(&[&pc], &pc_in_range.implies(&definition)),
		);
	}

	fn define_transition_stack(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = self
			.constants
			.in_range(&self.ctx.from_u64(0), &pc, &self.program_length);

		// constants
		let instr = self.program(&pc);
		let stack_pointer = self.stack_pointer(&pc);

		// encode instruction effect
		let new_stack_pointer = self.stack_pointer(&pc.add(&[&self.ctx.from_u64(1)]));

		// instr == Nop
		let definition = self
			.transition_stack(&pc, &self.constants.instruction(&Instruction::Nop))
			._eq(&self.ctx.from_bool(true));
		self.solver.assert(
			&self
				.ctx
				.forall_const(&[&pc], &pc_in_range.implies(&definition)),
		);

		// instr == Add implies stack(pc + 1, new_stack_pointer - 1) == stack(pc, stack_pointer - 1) + stack(pc, stack_pointer - 2)
		let sum = self
			.stack(&pc, &stack_pointer.sub(&[&self.ctx.from_i64(1)]))
			.bvadd(&self.stack(&pc, &stack_pointer.sub(&[&self.ctx.from_i64(2)])));
		let rhs = self
			.stack(
				&pc.add(&[&self.ctx.from_u64(1)]),
				&new_stack_pointer.sub(&[&self.ctx.from_i64(1)]),
			)
			._eq(&sum);
		let definition = self
			.transition_stack(&pc, &self.constants.instruction(&Instruction::I32Add))
			._eq(&rhs);
		self.solver.assert(
			&self
				.ctx
				.forall_const(&[&pc], &pc_in_range.implies(&definition)),
		);

		// instr == Const implies stack(pc + 1, new_stack_pointer - 1) == consts(pc)
		let rhs = self
			.stack(
				&pc.add(&[&self.ctx.from_u64(1)]),
				&new_stack_pointer.sub(&[&self.ctx.from_i64(1)]),
			)
			._eq(&self.push_constants(&pc));
		let definition = self
			.transition_stack(&pc, &self.constants.instruction(&Instruction::I32Const(0)))
			._eq(&rhs);
		self.solver.assert(
			&self
				.ctx
				.forall_const(&[&pc], &pc_in_range.implies(&definition)),
		);
	}

	fn define_transition(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = self
			.constants
			.in_range(&self.ctx.from_u64(0), &pc, &self.program_length);

		let definition = self.transition(&pc).iff(&self.preserve_stack(&pc).and(&[
			&self.transition_stack_pointer(&pc),
			&self.transition_stack(&pc, &self.program(&pc)),
		]));

		self.solver.assert(
			&self
				.ctx
				.forall_const(&[&pc], &pc_in_range.implies(&definition)),
		);
	}

	fn define_preserve_stack(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = self
			.constants
			.in_range(&self.ctx.from_u64(0), &pc, &self.program_length);

		// constants
		let instr = self.program(&pc);
		let stack_pointer = self.stack_pointer(&pc);

		// preserve stack values stack(_, 0)..=stack(_, stack_pointer - pops - 1)
		let n = self.ctx.named_int_const("n");

		let n_in_range = self.constants.in_range(
			&self.ctx.from_u64(0),
			&n,
			&stack_pointer.sub(&[&self.constants.stack_pop_count(&instr)]),
		);
		let slot_preserved = self
			.stack(&pc, &n)
			._eq(&self.stack(&pc.add(&[&self.ctx.from_u64(1)]), &n));
		let body = n_in_range.implies(&slot_preserved);

		// forall n
		let stack_is_preserved = self.ctx.forall_const(&[&n], &body);

		let definition = self.preserve_stack(&pc).iff(&stack_is_preserved);

		self.solver.assert(
			&self
				.ctx
				.forall_const(&[&pc], &pc_in_range.implies(&definition)),
		);
	}

	fn assert_transitions(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = self
			.constants
			.in_range(&self.ctx.from_u64(0), &pc, &self.program_length);

		let mut bounds: Vec<_> = self.constants.initial_stack.iter().collect();
		bounds.push(&pc);
		self.solver.assert(
			&self
				.ctx
				.forall_const(&bounds, &pc_in_range.implies(&self.transition(&pc))),
		);
	}

	fn transition(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.transition_func.apply(&[pc])
	}

	fn transition_stack_pointer(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.transition_stack_pointer_func.apply(&[pc])
	}

	fn transition_stack(&self, pc: &Ast<'ctx>, instr: &Ast<'ctx>) -> Ast<'ctx> {
		self.transition_stack_func.apply(&[pc, instr])
	}

	fn preserve_stack(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.preserve_stack_func.apply(&[pc])
	}

	fn stack_pointer(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.stack_pointer_func.apply(&[pc])
	}

	fn stack(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		self.stack_func.apply(&[pc, index])
	}

	fn program(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.program_func.apply(&[pc])
	}

	fn push_constants(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.push_constants_func.apply(&[pc])
	}
}

fn define_equivalent<'ctx>(lhs: &'ctx State, rhs: &'ctx State) -> FuncDecl<'ctx> {
	let ctx = lhs.ctx;
	let constants = rhs.constants;
	let solver = lhs.solver;

	let lhs_pc = ctx.named_int_const("lhs_pc");
	let rhs_pc = ctx.named_int_const("rhs_pc");

	let lhs_pc_in_range = constants.in_range(
		&ctx.from_u64(0),
		&lhs_pc,
		// +1 to allow querying for the final state
		&lhs.program_length.add(&[&ctx.from_u64(1)]),
	);
	let rhs_pc_in_range = constants.in_range(
		&ctx.from_u64(0),
		&rhs_pc,
		// +1 to allow querying for the final state
		&rhs.program_length.add(&[&ctx.from_u64(1)]),
	);
	let pcs_in_range = lhs_pc_in_range.and(&[&rhs_pc_in_range]);

	let name = lhs.prefix.to_owned() + &rhs.prefix + "equivalent";
	let equivalent_func = ctx.func_decl(
		ctx.str_sym(&name),
		&[&ctx.int_sort(), &ctx.int_sort()],
		&ctx.bool_sort(),
	);

	let stack_pointers_equal = lhs
		.stack_pointer_func
		.apply(&[&lhs_pc])
		._eq(&rhs.stack_pointer_func.apply(&[&rhs_pc]));

	let stacks_equal = {
		// for 0 <= n < stack_pointer
		let n = ctx.named_int_const("n");
		let n_in_range = constants.in_range(
			&ctx.from_u64(0),
			&n,
			&lhs.stack_pointer_func.apply(&[&lhs_pc]),
		);

		// lhs-stack(lhs_pc, n) ==  rhs-stack(rhs_pc, n)
		let condition = lhs
			.stack_func
			.apply(&[&lhs_pc, &n])
			._eq(&rhs.stack_func.apply(&[&rhs_pc, &n]));

		ctx.forall_const(&[&n], &n_in_range.implies(&condition))
	};

	let expected = equivalent_func.apply(&[&lhs_pc, &rhs_pc]);
	let actual = stack_pointers_equal.and(&[&stacks_equal]);
	let equal = expected._eq(&actual);
	solver.assert(&ctx.forall_const(&[&lhs_pc, &rhs_pc], &pcs_in_range.implies(&equal)));

	equivalent_func
}

fn gas_particle_cost(_: &Instruction) -> usize {
	1
}

pub fn superoptimize(source_program: &[Instruction]) -> Vec<Instruction> {
	let config = Config::default();
	let ctx = Context::new(&config);
	let solver = Solver::new(&ctx);

	let constants = Constants::new(&ctx, &solver, stack_depth(source_program) as usize);
	let source_state = State::new(&ctx, &solver, &constants, "source-");
	let target_state = State::new(&ctx, &solver, &constants, "target-");
	source_state.set_source_program(source_program);

	let source_len = source_program.len() as u64;
	let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

	source_state.assert_transitions();
	target_state.assert_transitions();

	let equivalent_func = define_equivalent(&source_state, &target_state);

	// start equivalent
	solver.assert(&equivalent_func.apply(&[&ctx.from_u64(0), &ctx.from_u64(0)]));

	let target_length = &target_state.program_length;

	let mut current_best = source_program.to_vec();

	loop {
		solver.push();

		// force target program to be shorter than current best
		solver.assert(&constants.in_range(
			&ctx.from_u64(0),
			target_length,
			&ctx.from_u64(current_best.len() as u64),
		));
		// assert programs are equivalent
		solver.assert(
			&equivalent_func.apply(&[&ctx.from_u64(source_program.len() as u64), &target_length]),
		);

		if !solver.check() {
			// already optimal
			return current_best;
		}

		// better version found
		// decode

		let model = solver.get_model();

		let target_length = model.eval(target_length).unwrap().as_i64().unwrap();
		let mut target_program = Vec::with_capacity(target_length as usize);

		for i in 0..target_length {
			let encoded_instr = model.eval(&target_state.program(&ctx.from_i64(i))).unwrap();

			for instr in iter_intructions() {
				let equal_tester = &constants.instruction_testers[instruction_to_index(instr)];
				let equal = model
					.eval(&equal_tester.apply(&[&encoded_instr]))
					.unwrap()
					.as_bool()
					.unwrap();

				if equal {
					let decoded = if let Instruction::I32Const(_) = instr {
						let push_constant_ast = target_state.push_constants(&ctx.from_i64(i));
						let push_constant =
							model.eval(&push_constant_ast).unwrap().as_i64().unwrap();
						// TODO fix cast
						Instruction::I32Const(push_constant as i32)
					} else {
						instr.clone()
					};

					target_program.push(decoded);

					break;
				}
			}
		}

		current_best = target_program;

		solver.pop(1);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_stack_pop_push_count() {
		let ctx = {
			let cfg = Config::default();
			Context::new(&cfg)
		};
		let solver = Solver::new(&ctx);
		let constants = Constants::new(&ctx, &solver, 0);

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: &Ast| -> i64 {
			let ast = model.eval(ast).unwrap();
			ast.as_i64().unwrap()
		};

		for i in iter_intructions() {
			let (pops, pushs) = stack_pop_push_count(i);
			let (pops, pushs) = (pops as i64, pushs as i64);
			assert_eq!(
				eval(&constants.stack_pop_count(&constants.instruction(i))),
				pops
			);
			assert_eq!(
				eval(&constants.stack_push_count(&constants.instruction(i))),
				pushs
			);
		}

		assert_eq!(
			eval(&constants.stack_pop_count(&constants.instruction(&Instruction::I32Add))),
			2
		);
		assert_eq!(
			eval(&constants.stack_push_count(&constants.instruction(&Instruction::I32Add))),
			1
		);
		assert_eq!(
			eval(&constants.stack_pop_count(&constants.instruction(&Instruction::I32Const(0)))),
			0
		);
		assert_eq!(
			eval(&constants.stack_push_count(&constants.instruction(&Instruction::I32Const(0)))),
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
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::Nop,
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		for (i, instr) in program.iter().enumerate() {
			let instr_enc = state.program(&ctx.from_u64(i as u64));
			let is_equal =
				constants.instruction_testers[instruction_to_index(instr)].apply(&[&instr_enc]);
			let b = model.eval(&is_equal).unwrap().as_bool().unwrap();
			assert!(b);
		}
	}

	#[test]
	fn initial_conditions() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		let stack_pointer = state.stack_pointer(&ctx.from_u64(0));
		let stack_pointer = model.eval(&stack_pointer).unwrap().as_i64().unwrap();
		assert_eq!(stack_pointer, 0);
	}

	#[test]
	fn stack_pointer_transition() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: Ast| -> i64 {
			let evaled = model.eval(&ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval(state.stack_pointer(&ctx.from_u64(0))), 0);
		assert_eq!(eval(state.stack_pointer(&ctx.from_u64(1))), 1);
		assert_eq!(eval(state.stack_pointer(&ctx.from_u64(2))), 2);
		assert_eq!(eval(state.stack_pointer(&ctx.from_u64(3))), 1);
	}

	#[test]
	fn consts() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Const(1), Instruction::I32Const(2)];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: Ast| -> i64 {
			let evaled = model.eval(&ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval(state.push_constants(&ctx.from_u64(0))), 1);
		assert_eq!(eval(state.push_constants(&ctx.from_u64(1))), 2);
	}

	#[test]
	fn transition_const() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval_int(&state.stack_pointer(&ctx.from_u64(1))), 1);

		let eval_bv = |ast: &Ast| -> i64 {
			let evaled = model.eval(&ast.bv2int(true)).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval_bv(&state.stack(&ctx.from_u64(1), &ctx.from_u64(0))), 1);
	}

	#[test]
	fn transition_add_consts() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::Nop,
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast).unwrap();
			evaled.as_i64().unwrap()
		};

		let eval_bv = |ast: &Ast| -> i64 {
			let evaled = model.eval(&ast.bv2int(true)).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval_bv(&state.stack(&ctx.from_u64(1), &ctx.from_u64(0))), 1);
		assert_eq!(eval_bv(&state.stack(&ctx.from_u64(2), &ctx.from_u64(0))), 1);
		assert_eq!(eval_bv(&state.stack(&ctx.from_u64(3), &ctx.from_u64(0))), 1);
		assert_eq!(eval_bv(&state.stack(&ctx.from_u64(3), &ctx.from_u64(1))), 2);
		assert_eq!(eval_bv(&state.stack(&ctx.from_u64(4), &ctx.from_u64(0))), 3);
	}

	#[test]
	fn transition_add() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Add];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast).unwrap();
			evaled.as_i64().unwrap()
		};

		let eval_bv = |ast: &Ast| -> i64 {
			let evaled = model.eval(&ast.bv2int(true)).unwrap();
			evaled.as_i64().unwrap()
		};
		let sum = constants.initial_stack[0].bvadd(&constants.initial_stack[1]);
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_u64(1), &ctx.from_u64(0))),
			eval_bv(&sum)
		);
	}

	#[test]
	fn equivalent_reflexive() {
		let config = Config::default();
		let ctx = Context::new(&config);
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::Nop,
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _);
		let source_state = State::new(&ctx, &solver, &constants, "source-");
		let target_state = State::new(&ctx, &solver, &constants, "target-");

		source_state.set_source_program(program);
		target_state.set_source_program(program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		source_state.assert_transitions();
		target_state.assert_transitions();

		let equivalent_func = define_equivalent(&source_state, &target_state);

		assert!(solver.check());
		let model = solver.get_model();

		let equivalent = |i, j| equivalent_func.apply(&[&ctx.from_u64(i), &ctx.from_u64(j)]);

		for i in 0..program.len() {
			let s = equivalent(i as u64, i as u64);
			let evaled = model.eval(&s).unwrap();
			let equiv = evaled.as_bool().unwrap();
			assert!(equiv);
		}
	}

	#[test]
	fn equivalent_nop_no_effect() {
		let config = Config::default();
		let ctx = Context::new(&config);
		let solver = Solver::new(&ctx);

		let lhs_program = &[Instruction::I32Const(1), Instruction::Nop];
		let rhs_program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(lhs_program) as _);
		let lhs_state = State::new(&ctx, &solver, &constants, "source-");
		let rhs_state = State::new(&ctx, &solver, &constants, "target-");
		lhs_state.set_source_program(lhs_program);
		rhs_state.set_source_program(rhs_program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		lhs_state.assert_transitions();
		rhs_state.assert_transitions();

		let equivalent_func = define_equivalent(&lhs_state, &rhs_state);

		assert!(solver.check());
		let model = solver.get_model();

		let s = equivalent_func.apply(&[&ctx.from_u64(2), &ctx.from_u64(1)]);
		let evaled = model.eval(&s).unwrap();
		let equiv = evaled.as_bool().unwrap();
		assert!(equiv);
	}

	#[test]
	fn equivalent_stack_unequal() {
		let config = Config::default();
		let ctx = Context::new(&config);
		let solver = Solver::new(&ctx);

		let lhs_program = &[Instruction::I32Const(1), Instruction::I32Const(2)];
		let rhs_program = &[Instruction::I32Const(2), Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(lhs_program) as _);
		let lhs_state = State::new(&ctx, &solver, &constants, "source-");
		let rhs_state = State::new(&ctx, &solver, &constants, "target-");
		lhs_state.set_source_program(lhs_program);
		rhs_state.set_source_program(rhs_program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		lhs_state.assert_transitions();
		rhs_state.assert_transitions();

		let equivalent_func = define_equivalent(&lhs_state, &rhs_state);

		assert!(solver.check());
		let model = solver.get_model();

		let s = equivalent_func.apply(&[&ctx.from_u64(2), &ctx.from_u64(2)]);
		let evaled = model.eval(&s).unwrap();
		let equiv = evaled.as_bool().unwrap();
		assert!(!equiv);
	}

	#[test]
	fn equivalent_stack_pointer_unequal() {
		let config = Config::default();
		let ctx = Context::new(&config);
		let solver = Solver::new(&ctx);

		let lhs_program = &[Instruction::I32Const(1), Instruction::I32Const(2)];
		let rhs_program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(lhs_program) as _);
		let lhs_state = State::new(&ctx, &solver, &constants, "source-");
		let rhs_state = State::new(&ctx, &solver, &constants, "target-");
		lhs_state.set_source_program(lhs_program);
		rhs_state.set_source_program(rhs_program);
		let initial_stack: Vec<_> = constants.initial_stack.iter().collect();

		lhs_state.assert_transitions();
		rhs_state.assert_transitions();

		let equivalent_func = define_equivalent(&lhs_state, &rhs_state);

		assert!(solver.check());
		let model = solver.get_model();

		let s = equivalent_func.apply(&[&ctx.from_u64(2), &ctx.from_u64(1)]);
		let evaled = model.eval(&s).unwrap();
		let equiv = evaled.as_bool().unwrap();
		assert!(!equiv);
	}

	#[test]
	fn superoptimize_nop() {
		let source_program = &[Instruction::I32Const(1), Instruction::Nop];
		let target = superoptimize(source_program);
		assert_eq!(target, vec![Instruction::I32Const(1)]);
	}

	#[test]
	fn superoptimize_consts_add() {
		let source_program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];
		let target = superoptimize(source_program);
		assert_eq!(target, vec![Instruction::I32Const(3)]);
	}

	#[test]
	fn superoptimize_add() {
		let source_program = &[Instruction::I32Const(0), Instruction::I32Add];
		let target = superoptimize(source_program);
		assert_eq!(target, vec![]);
	}
}
