use crate::instructions::*;
use crate::*;
use parity_wasm::elements::Instruction;
use z3::*;

pub struct State<'ctx, 'solver, 'constants> {
	pub ctx: &'ctx Context,
	pub solver: &'solver Solver<'ctx>,
	pub constants: &'constants Constants<'ctx>,
	pub prefix: String,
}

impl<'ctx, 'solver, 'constants> State<'ctx, 'solver, 'constants> {
	pub fn new(
		ctx: &'ctx Context,
		solver: &'solver Solver<'ctx>,
		constants: &'constants Constants<'ctx>,
		prefix: &str,
	) -> Self {
		let state = State {
			ctx,
			solver,
			constants,
			prefix: prefix.to_string(),
		};

		state.set_initial();

		state
	}

	pub fn set_initial(&self) {
		// set stack(0, i) == initial_stack[i]
		for (i, var) in self.constants.initial_stack.iter().enumerate() {
			self.solver.assert(
				&self
					.stack(&self.ctx.from_usize(0), &self.ctx.from_usize(i))
					._eq(&var),
			);
		}

		// set stack_counter(0) = 0
		self.solver.assert(
			&self
				.stack_pointer(&self.ctx.from_usize(0))
				._eq(&self.ctx.from_usize(self.constants.stack_depth)),
		);

		// set params
		for (i, var) in self.constants.params.iter().enumerate() {
			self.solver.assert(
				&self
					.local(&self.ctx.from_usize(0), &self.ctx.from_usize(i))
					._eq(&var),
			);
		}

		// force n_locals to be >= n_params
		let n_params = self.ctx.from_usize(self.constants.params.len());
		self.solver.assert(&self.n_locals().ge(&n_params));

		// set remaining locals to 0
		let n = self.ctx.named_int_const("n");
		let bv_zero = self.ctx.from_usize(0).int2bv(32);
		let n_in_range = in_range(&n_params, &n, &self.n_locals());
		self.solver.assert(&self.ctx.forall_const(
			&[&n],
			&n_in_range.implies(&self.local(&self.ctx.from_usize(0), &n)._eq(&bv_zero)),
		));

		// constrain 0 <= local_index <= n_locals
		let pc = self.ctx.named_int_const("pc");
		let local_index_in_range = in_range(
			&self.ctx.from_usize(0),
			&self.local_index(&pc),
			&self.n_locals(),
		);
		self.solver
			.assert(&self.ctx.forall_const(&[&pc], &local_index_in_range));
	}

	pub fn set_source_program(&self, program: &[Instruction]) {
		// set program_func to program
		for (pc, instruction) in program.iter().enumerate() {
			let instruction = self.constants.instruction(instruction);
			self.solver
				.assert(&self.program(&self.ctx.from_usize(pc))._eq(&instruction))
		}

		for (pc, instr) in program.iter().enumerate() {
			use Instruction::*;
			let pc = self.ctx.from_usize(pc);

			// set push_constants function
			match instr {
				I32Const(i) => {
					let i = self.ctx.from_i32(*i).int2bv(32);
					self.solver.assert(&self.push_constants(&pc)._eq(&i));
				}
				GetLocal(i) | SetLocal(i) | TeeLocal(i) => {
					self.solver
						.assert(&self.local_index(&pc)._eq(&self.ctx.from_u32(*i)));
				}
				_ => {}
			}
		}

		// set length
		self.solver.assert(
			&self
				.program_length()
				._eq(&self.ctx.from_usize(program.len())),
		);
	}

	pub fn assert_transitions(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = in_range(&self.ctx.from_usize(0), &pc, &self.program_length());

		// forall initial_stack values and all params and all pcs
		let mut bounds: Vec<_> = self
			.constants
			.initial_stack
			.iter()
			.chain(self.constants.params.iter())
			.collect();
		bounds.push(&pc);
		let transition = self.transition(&pc);
		self.solver.assert(
			&self
				.ctx
				.forall_const(&bounds, &pc_in_range.implies(&transition)),
		);
	}

	fn transition(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.ctx.from_bool(true).and(&[
			&self.preserve_stack(&pc),
			&self.preserve_locals(&pc),
			&self.transition_stack_pointer(&pc),
			&self.transition_stack(&pc),
		])
	}

	fn transition_stack_pointer(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		// encode stack_pointer change
		let stack_pointer = self.stack_pointer(&pc);
		let stack_pointer_next = self.stack_pointer(&pc.add(&[&self.ctx.from_usize(1)]));

		let instruction = self.program(&pc);
		let pop_count = self.constants.stack_pop_count(&instruction);
		let push_count = self.constants.stack_push_count(&instruction);

		let new_pointer = stack_pointer.add(&[&push_count]).sub(&[&pop_count]);

		stack_pointer_next._eq(&new_pointer)
	}

	fn transition_stack(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		use Instruction::*;

		let instr = self.program(&pc);

		let transition_instruction = |i: &Instruction, ast: &Ast<'ctx>| -> Ast<'ctx> {
			self.constants.instruction(i)._eq(&instr).implies(ast)
		};

		// ad-hoc conversions
		let bool_to_i32 = |b: &Ast<'ctx>| {
			b.ite(
				&self.ctx.from_usize(1).int2bv(32),
				&self.ctx.from_usize(0).int2bv(32),
			)
		};
		let mod_n = |b: &Ast<'ctx>, n: usize| b.bvurem(&self.ctx.from_usize(n).int2bv(32));

		// constants
		let bv_zero = self.ctx.from_usize(0).int2bv(32);
		let pc_next = &pc.add(&[&self.ctx.from_usize(1)]);

		let op1 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(1)]));
		let op2 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(2)]));
		let op3 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(3)]));
		let result = self.stack(
			pc_next,
			&self.stack_pointer(&pc_next).sub(&[&self.ctx.from_i64(1)]),
		);
		let current_local = self.local(&pc, &self.local_index(&pc));
		// local_index still with pc
		let next_local = self.local(&pc_next, &self.local_index(&pc));

		let transitions = &[
			// Nop: no semantics
			&transition_instruction(&I32Const(0), &result._eq(&self.push_constants(&pc))),
			&transition_instruction(&I32Eqz, &result._eq(&bool_to_i32(&op1._eq(&bv_zero)))),
			&transition_instruction(&I32Eq, &result._eq(&bool_to_i32(&op2._eq(&op1)))),
			&transition_instruction(&I32Ne, &result._eq(&bool_to_i32(&op2._eq(&op1).not()))),
			&transition_instruction(&I32LtS, &result._eq(&bool_to_i32(&op2.bvslt(&op1)))),
			&transition_instruction(&I32LtU, &result._eq(&bool_to_i32(&op2.bvult(&op1)))),
			&transition_instruction(&I32GtS, &result._eq(&bool_to_i32(&op2.bvsgt(&op1)))),
			&transition_instruction(&I32GtU, &result._eq(&bool_to_i32(&op2.bvugt(&op1)))),
			&transition_instruction(&I32LeS, &result._eq(&bool_to_i32(&op2.bvsle(&op1)))),
			&transition_instruction(&I32LeU, &result._eq(&bool_to_i32(&op2.bvule(&op1)))),
			&transition_instruction(&I32GeS, &result._eq(&bool_to_i32(&op2.bvsge(&op1)))),
			&transition_instruction(&I32GeU, &result._eq(&bool_to_i32(&op2.bvuge(&op1)))),
			// TODO
			// I32Clz
			// I32Ctz
			// I32Popcnt
			&transition_instruction(&I32Add, &result._eq(&op2.bvadd(&op1))),
			&transition_instruction(&I32Sub, &result._eq(&op2.bvsub(&op1))),
			&transition_instruction(&I32Mul, &result._eq(&op2.bvmul(&op1))),
			&transition_instruction(&I32DivS, &result._eq(&op2.bvsdiv(&op1))),
			&transition_instruction(&I32DivU, &result._eq(&op2.bvudiv(&op1))),
			&transition_instruction(&I32RemS, &result._eq(&op2.bvsrem(&op1))),
			&transition_instruction(&I32RemU, &result._eq(&op2.bvurem(&op1))),
			&transition_instruction(&I32And, &result._eq(&op2.bvand(&op1))),
			&transition_instruction(&I32Or, &result._eq(&op2.bvor(&op1))),
			&transition_instruction(&I32Xor, &result._eq(&op2.bvxor(&op1))),
			&transition_instruction(&I32Shl, &result._eq(&op2.bvshl(&mod_n(&op1, 32)))),
			&transition_instruction(&I32ShrS, &result._eq(&op2.bvashr(&mod_n(&op1, 32)))),
			&transition_instruction(&I32ShrU, &result._eq(&op2.bvlshr(&mod_n(&op1, 32)))),
			&transition_instruction(&I32Rotl, &result._eq(&op2.bvrotl(&mod_n(&op1, 32)))),
			&transition_instruction(&I32Rotr, &result._eq(&op2.bvrotr(&mod_n(&op1, 32)))),
			// Drop: no semantics
			&transition_instruction(&Select, &result._eq(&op1._eq(&bv_zero).ite(&op2, &op3))),
			// locals
			&transition_instruction(&GetLocal(0), &result._eq(&current_local)),
			// pop count is different between SetLocal and TeeLocal
			&transition_instruction(&SetLocal(0), &next_local._eq(&op1)),
			&transition_instruction(&TeeLocal(0), &next_local._eq(&op1)),
		];
		self.ctx.from_bool(true).and(transitions)
	}

	fn preserve_stack(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		// constants
		let instr = self.program(&pc);
		let stack_pointer = self.stack_pointer(&pc);

		// preserve stack values stack(_, 0)..=stack(_, stack_pointer - pops - 1)
		let n = self.ctx.named_int_const("n");

		let n_in_range = in_range(
			&self.ctx.from_usize(0),
			&n,
			&stack_pointer.sub(&[&self.constants.stack_pop_count(&instr)]),
		);
		let slot_preserved = self
			.stack(&pc, &n)
			._eq(&self.stack(&pc.add(&[&self.ctx.from_usize(1)]), &n));

		// forall n
		self.ctx
			.forall_const(&[&n], &n_in_range.implies(&slot_preserved))
	}

	fn preserve_locals(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		// preserve all locals which are not set in this step
		let i = self.ctx.named_int_const("i");
		let i_in_range = in_range(&self.ctx.from_usize(0), &i, &self.n_locals());

		let is_setlocal = self.constants.instruction_testers
			[instruction_to_index(&Instruction::SetLocal(0))]
		.apply(&[&self.program(&pc)]);
		let is_teelocal = self.constants.instruction_testers
			[instruction_to_index(&Instruction::TeeLocal(0))]
		.apply(&[&self.program(&pc)]);
		let is_setting_instruction = is_setlocal.or(&[&is_teelocal]);
		let index_active = i._eq(&self.local_index(&pc));
		let enable = is_setting_instruction.and(&[&index_active]).not();

		let pc_next = pc.add(&[&self.ctx.from_usize(1)]);

		self.ctx.forall_const(
			&[&i],
			&i_in_range
				.and(&[&enable])
				.implies(&self.local(&pc_next, &i)._eq(&self.local(&pc, &i))),
		)
	}

	// stack_pointer - 1 is top of stack
	pub fn stack_pointer(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_pointer_func = self.ctx.func_decl(
			self.ctx
				.str_sym(&(self.prefix.to_owned() + "stack_pointer")),
			&[&self.ctx.int_sort()],
			&self.ctx.int_sort(),
		);

		stack_pointer_func.apply(&[pc])
	}

	pub fn stack(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "stack")),
			&[
				&self.ctx.int_sort(), // instruction counter
				&self.ctx.int_sort(), // stack address
			],
			&self.ctx.bitvector_sort(32),
		);

		stack_func.apply(&[pc, index])
	}

	pub fn program(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let program_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "program")),
			&[&self.ctx.int_sort()],
			&self.constants.instruction_sort,
		);

		program_func.apply(&[pc])
	}

	pub fn push_constants(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let push_constants_func = self.ctx.func_decl(
			self.ctx
				.str_sym(&(self.prefix.to_owned() + "push_constants")),
			&[&self.ctx.int_sort()],
			&self.ctx.bitvector_sort(32),
		);

		push_constants_func.apply(&[pc])
	}

	pub fn local(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let local_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "local")),
			&[&self.ctx.int_sort(), &self.ctx.int_sort()],
			&self.ctx.bitvector_sort(32),
		);

		local_func.apply(&[pc, index])
	}

	pub fn local_index(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let local_index_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "local_index")),
			&[&self.ctx.int_sort()],
			&self.ctx.int_sort(),
		);

		local_index_func.apply(&[pc])
	}

	pub fn program_length(&self) -> Ast<'ctx> {
		self.ctx
			.named_int_const(&(self.prefix.to_owned() + "program_length"))
	}

	// number of locals including params
	pub fn n_locals(&self) -> Ast<'ctx> {
		self.ctx
			.named_int_const(&(self.prefix.to_owned() + "n_locals"))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

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

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		for (i, instr) in program.iter().enumerate() {
			let instr_enc = state.program(&ctx.from_usize(i));
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		let stack_pointer = state.stack_pointer(&ctx.from_usize(0));
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: Ast| -> i64 {
			let evaled = model.eval(&ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval(state.stack_pointer(&ctx.from_usize(0))), 0);
		assert_eq!(eval(state.stack_pointer(&ctx.from_usize(1))), 1);
		assert_eq!(eval(state.stack_pointer(&ctx.from_usize(2))), 2);
		assert_eq!(eval(state.stack_pointer(&ctx.from_usize(3))), 1);
	}

	#[test]
	fn consts() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Const(1), Instruction::I32Const(2)];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: Ast| -> i64 {
			let evaled = model.eval(&ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval(state.push_constants(&ctx.from_usize(0))), 1);
		assert_eq!(eval(state.push_constants(&ctx.from_usize(1))), 2);
	}

	#[test]
	fn transition_const() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Const(1)];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval_int(&state.stack_pointer(&ctx.from_usize(1))), 1);

		let eval_bv = |ast: &Ast| -> i64 {
			let evaled = model.eval(&ast.bv2int(true)).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(1), &ctx.from_usize(0))),
			1
		);
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval_bv = |ast: &Ast| -> i64 {
			let evaled = model.eval(&ast.bv2int(true)).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(1), &ctx.from_usize(0))),
			1
		);
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(2), &ctx.from_usize(0))),
			1
		);
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(3), &ctx.from_usize(0))),
			1
		);
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(3), &ctx.from_usize(1))),
			2
		);
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(4), &ctx.from_usize(0))),
			3
		);
	}

	#[test]
	fn transition_add() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Add];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		let eval_bv = |ast: &Ast| -> i64 {
			let evaled = model.eval(&ast.bv2int(true)).unwrap();
			evaled.as_i64().unwrap()
		};
		let sum = constants.initial_stack[0].bvadd(&constants.initial_stack[1]);
		assert_eq!(
			eval_bv(&state.stack(&ctx.from_usize(1), &ctx.from_usize(0))),
			eval_bv(&sum)
		);
	}

	#[test]
	fn transition_const_drop() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Const(1), Instruction::Drop];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model
				.eval(&state.stack_pointer(&ctx.from_usize(2)))
				.unwrap()
				.as_usize()
				.unwrap(),
			0
		);
	}

	#[test]
	fn transition_consts_select_true() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Const(3),
			Instruction::Select,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model
				.eval(&state.stack_pointer(&ctx.from_usize(4)))
				.unwrap()
				.as_usize()
				.unwrap(),
			1
		);
		assert_eq!(
			model
				.eval(
					&state
						.stack(&ctx.from_usize(4), &ctx.from_usize(0))
						.bv2int(false)
				)
				.unwrap()
				.as_usize()
				.unwrap(),
			1
		);
	}

	#[test]
	fn transition_consts_select_false() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Const(0),
			Instruction::Select,
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 0);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model
				.eval(&state.stack_pointer(&ctx.from_usize(4)))
				.unwrap()
				.as_usize()
				.unwrap(),
			1
		);
		assert_eq!(
			model
				.eval(
					&state
						.stack(&ctx.from_usize(4), &ctx.from_usize(0))
						.bv2int(false)
				)
				.unwrap()
				.as_usize()
				.unwrap(),
			2
		);
	}

	#[test]
	fn transition_local() {
		use Instruction::*;

		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			// x2 = x1 + x0
			GetLocal(0),
			GetLocal(1),
			I32Add,
			SetLocal(2),
			// swap x0, x1
			// tmp = x1; x1 = x0; x0 = tmp;
			GetLocal(1),
			GetLocal(0),
			SetLocal(1),
			SetLocal(0),
		];

		let constants = Constants::new(&ctx, &solver, stack_depth(program), 2);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();
		constants.set_params(&solver, &[1, 2]);

		println!("{}", &solver);

		assert!(solver.check());
		let model = solver.get_model();

		println!("{}", &model);

		assert_eq!(
			model.eval(&state.n_locals()).unwrap().as_usize().unwrap(),
			3
		);

		let stack_pointer = |pc| {
			model
				.eval(&state.stack_pointer(&ctx.from_usize(pc)))
				.unwrap()
				.as_usize()
				.unwrap()
		};
		let stack = |pc, i| {
			model
				.eval(
					&state
						.stack(&ctx.from_usize(pc), &ctx.from_usize(i))
						.bv2int(false),
				)
				.unwrap()
				.as_usize()
				.unwrap()
		};
		let local = |pc, i| {
			model
				.eval(
					&state
						.local(&ctx.from_usize(pc), &ctx.from_usize(i))
						.bv2int(false),
				)
				.unwrap()
				.as_usize()
				.unwrap()
		};

		assert_eq!(local(0, 0), 1);
		assert_eq!(local(0, 1), 2);
		// default value
		assert_eq!(local(0, 2), 0);

		assert_eq!(stack_pointer(1), 1);
		assert_eq!(stack(1, 0), 1);
		assert_eq!(stack_pointer(2), 2);
		assert_eq!(stack(2, 1), 2);

		assert_eq!(stack_pointer(4), 0);
		assert_eq!(local(4, 2), 3);

		assert_eq!(stack_pointer(8), 0);
		assert_eq!(local(8, 0), 2);
		assert_eq!(local(8, 1), 1);
	}
}
