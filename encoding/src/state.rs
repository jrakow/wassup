use crate::instructions::{instruction_datatype, Instruction};
use crate::*;
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
		// set trapped(0) = false
		self.solver
			.assert(&self.trapped(&self.ctx.from_usize(0)).not());

		// set stack(0, i) == initial_stack[i]
		for (i, var) in self.constants.initial_stack.iter().enumerate() {
			self.solver.assert(
				&self
					.stack(&self.ctx.from_usize(0), &self.ctx.from_usize(i))
					._eq(&var),
			);
		}

		// set stack_counter(0) = initial_stack.len()
		self.solver.assert(
			&self
				.stack_pointer(&self.ctx.from_usize(0))
				._eq(&self.ctx.from_usize(self.constants.initial_stack.len())),
		);

		// set n_locals = initial_locals.len()
		let n_locals = self.ctx.from_usize(self.constants.initial_locals.len());
		self.solver.assert(&self.n_locals()._eq(&n_locals));

		// set local(0, i) = inital_locals[i]
		for (i, var) in self.constants.initial_locals.iter().enumerate() {
			self.solver.assert(
				&self
					.local(&self.ctx.from_usize(0), &self.ctx.from_usize(i))
					._eq(&var),
			);
		}

		// constrain 0 <= local_index < n_locals
		let pc = self.ctx.named_int_const("pc");
		let instr = self.program(&pc);
		let mut conditions = Vec::new();
		for i in &[
			Instruction::I32GetLocal(0),
			Instruction::I32SetLocal(0),
			Instruction::I32TeeLocal(0),
		] {
			let variant = &instruction_datatype(self.ctx).variants[i.as_usize()];
			let active = variant.tester.apply(&[&instr]);
			let index = variant.accessors[0].apply(&[&instr]);

			let index_in_range = in_range(&self.ctx.from_usize(0), &index, &self.n_locals());

			conditions.push(active.implies(&index_in_range));
		}
		// as_ref
		let conditions: Vec<_> = conditions.iter().collect();
		let combined = self.ctx.from_bool(true).and(&conditions);
		self.solver
			.assert(&self.ctx.forall_const(&[&pc], &combined));
	}

	pub fn set_source_program(&self, program: &[Instruction]) {
		for (pc, instruction) in program.iter().enumerate() {
			let pc = self.ctx.from_usize(pc);

			// set program
			self.solver
				.assert(&self.program(&pc)._eq(&instruction.encode(self.ctx)));
		}

		// set program_length
		self.solver.assert(
			&self
				.program_length()
				._eq(&self.ctx.from_usize(program.len())),
		);
	}

	pub fn decode_program(&self, model: &Model) -> Vec<Instruction> {
		use Instruction::*;

		let program_length = model
			.eval(&self.program_length())
			.unwrap()
			.as_usize()
			.unwrap();

		let mut program = Vec::with_capacity(program_length);

		for pc in 0..program_length {
			let pc = self.ctx.from_usize(pc);
			let encoded_instr = model.eval(&self.program(&pc)).unwrap();

			let decoded = Instruction::iter_templates()
				.find_map(|template| {
					let variant = &instruction_datatype(self.ctx).variants[template.as_usize()];

					let active = variant.tester.apply(&[&encoded_instr]);
					if model.eval(&active).unwrap().as_bool().unwrap() {
						Some(match template {
							I32Const(_) => {
								let ast = variant.accessors[0].apply(&[&encoded_instr]);
								let v = model.eval(&ast.bv2int(true)).unwrap().as_i32().unwrap();
								I32Const(v)
							}
							I32GetLocal(_) => {
								let ast = variant.accessors[0].apply(&[&encoded_instr]);
								let v = model.eval(&ast).unwrap().as_u32().unwrap();
								I32GetLocal(v)
							}
							I32SetLocal(_) => {
								let ast = variant.accessors[0].apply(&[&encoded_instr]);
								let v = model.eval(&ast).unwrap().as_u32().unwrap();
								I32SetLocal(v)
							}
							I32TeeLocal(_) => {
								let ast = variant.accessors[0].apply(&[&encoded_instr]);
								let v = model.eval(&ast).unwrap().as_u32().unwrap();
								I32TeeLocal(v)
							}
							x => x,
						})
					} else {
						None
					}
				})
				.unwrap();

			program.push(decoded);
		}

		program
	}

	pub fn transitions(&self) -> Ast<'ctx> {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = in_range(&self.ctx.from_usize(0), &pc, &self.program_length());

		// forall initial_stack values and all params and all pcs
		let mut bounds: Vec<_> = self.constants.initial_stack.iter().collect();
		bounds.push(&pc);
		let transition = self.transition(&pc);
		self.ctx
			.forall_const(&bounds, &pc_in_range.implies(&transition))
	}

	fn transition(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		self.ctx.from_bool(true).and(&[
			&self.preserve_stack(&pc),
			&self.preserve_locals(&pc),
			&self.transition_stack_pointer(&pc),
			&self.transition_stack(&pc),
			&self.transition_trapped(&pc),
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

		let instruction_datatype = instruction_datatype(self.ctx);
		let instr = self.program(&pc);

		// ad-hoc conversions
		let bool_to_i32 = |b: &Ast<'ctx>| {
			b.ite(
				&self.ctx.from_usize(1).int2bv(32),
				&self.ctx.from_usize(0).int2bv(32),
			)
		};
		let mod_n = |b: &Ast<'ctx>, n: usize| b.bvurem(&self.ctx.from_usize(n).int2bv(32));

		// constants
		let zero = self.ctx.from_usize(0);
		let bv_zero = self.ctx.from_usize(0).int2bv(32);
		let pc_next = &pc.add(&[&self.ctx.from_usize(1)]);

		let op1 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(1)]));
		let op2 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(2)]));
		let op3 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(3)]));
		let result = self.stack(
			pc_next,
			&self.stack_pointer(&pc_next).sub(&[&self.ctx.from_i64(1)]),
		);

		let mut transitions = Vec::new();
		for (i, variant) in instruction_datatype.variants.iter().enumerate() {
			let active = variant.tester.apply(&[&instr]);

			let transition = match Instruction::iter_templates().nth(i).unwrap() {
				Unreachable | Nop | I32Drop => self.ctx.from_bool(true),

				I32Const(_) => result._eq(&variant.accessors[0].apply(&[&instr])),

				I32Eqz => result._eq(&bool_to_i32(&op1._eq(&bv_zero))),
				I32Eq => result._eq(&bool_to_i32(&op2._eq(&op1))),
				I32Ne => result._eq(&bool_to_i32(&op2._eq(&op1).not())),
				I32LtS => result._eq(&bool_to_i32(&op2.bvslt(&op1))),
				I32LtU => result._eq(&bool_to_i32(&op2.bvult(&op1))),
				I32GtS => result._eq(&bool_to_i32(&op2.bvsgt(&op1))),
				I32GtU => result._eq(&bool_to_i32(&op2.bvugt(&op1))),
				I32LeS => result._eq(&bool_to_i32(&op2.bvsle(&op1))),
				I32LeU => result._eq(&bool_to_i32(&op2.bvule(&op1))),
				I32GeS => result._eq(&bool_to_i32(&op2.bvsge(&op1))),
				I32GeU => result._eq(&bool_to_i32(&op2.bvuge(&op1))),

				I32Add => result._eq(&op2.bvadd(&op1)),
				I32Sub => result._eq(&op2.bvsub(&op1)),
				I32Mul => result._eq(&op2.bvmul(&op1)),
				I32DivS => result._eq(&op2.bvsdiv(&op1)),
				I32DivU => result._eq(&op2.bvudiv(&op1)),
				I32RemS => result._eq(&op2.bvsrem(&op1)),
				I32RemU => result._eq(&op2.bvurem(&op1)),
				I32And => result._eq(&op2.bvand(&op1)),
				I32Or => result._eq(&op2.bvor(&op1)),
				I32Xor => result._eq(&op2.bvxor(&op1)),
				I32Shl => result._eq(&op2.bvshl(&mod_n(&op1, 32))),
				I32ShrS => result._eq(&op2.bvashr(&mod_n(&op1, 32))),
				I32ShrU => result._eq(&op2.bvlshr(&mod_n(&op1, 32))),
				I32Rotl => result._eq(&op2.bvrotl(&mod_n(&op1, 32))),
				I32Rotr => result._eq(&op2.bvrotr(&mod_n(&op1, 32))),

				I32Select => result._eq(&op1._eq(&bv_zero).ite(&op2, &op3)),

				// SetLocal and TeeLocal differ in pop count
				I32GetLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);
					let index_in_range = in_range(&zero, &index, &self.n_locals());
					result._eq(&self.local(&pc, &index)).and(&[&index_in_range])
				}
				I32SetLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);
					let index_in_range = in_range(&zero, &index, &self.n_locals());
					self.local(&pc_next, &index)
						._eq(&op1)
						.and(&[&index_in_range])
				}
				I32TeeLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);

					let index_in_range = in_range(&zero, &index, &self.n_locals());
					let local_set = self.local(&pc_next, &index)._eq(&op1);
					let stack_set = result._eq(&op1);

					self.ctx
						.from_bool(true)
						.and(&[&index_in_range, &local_set, &stack_set])
				}
			};
			transitions.push(active.implies(&transition));
		}

		// create vector of references
		let transitions: Vec<_> = transitions.iter().collect();
		self.ctx.from_bool(true).and(&transitions)
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
		let instr = self.program(&pc);

		// preserve all locals which are not set in this step
		let i = self.ctx.named_int_const("i");
		let i_in_range = in_range(&self.ctx.from_usize(0), &i, &self.n_locals());

		let variants = &instruction_datatype(self.ctx).variants;

		// disable if set_local
		let set_local = &variants[Instruction::I32SetLocal(0).as_usize()];
		let is_set_local = set_local.tester.apply(&[&instr]);
		let set_local_index = set_local.accessors[0].apply(&[&instr]);
		let set_local_active = is_set_local.and(&[&set_local_index._eq(&i)]);

		let tee_local = &variants[Instruction::I32TeeLocal(0).as_usize()];
		let is_tee_local = tee_local.tester.apply(&[&instr]);
		let tee_local_index = tee_local.accessors[0].apply(&[&instr]);
		let tee_local_active = is_tee_local.and(&[&tee_local_index._eq(&i)]);

		let pc_next = pc.add(&[&self.ctx.from_usize(1)]);

		self.ctx.forall_const(
			&[&i],
			&i_in_range
				.and(&[&set_local_active.not(), &tee_local_active.not()])
				.implies(&self.local(&pc_next, &i)._eq(&self.local(&pc, &i))),
		)
	}

	fn transition_trapped(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		use Instruction::*;

		let instr = self.program(&pc);
		let pc_next = pc.add(&[&self.ctx.from_usize(1)]);

		let bv_zero = self.ctx.from_usize(0).int2bv(32);

		let op1 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(1)]));
		let op2 = self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(2)]));

		let mut conditions = Vec::new();
		conditions.push(self.trapped(&pc));

		for (i, variant) in instruction_datatype(self.ctx).variants.iter().enumerate() {
			let active = variant.tester.apply(&[&instr]);

			let condition = match Instruction::iter_templates().nth(i).unwrap() {
				Unreachable => self.ctx.from_bool(true),

				I32DivU | I32RemU | I32RemS => op2._eq(&bv_zero),
				I32DivS => {
					let divide_by_zero = op2._eq(&bv_zero);
					let overflow = op2.bvsdiv_no_overflow(&op1);
					divide_by_zero.or(&[&overflow])
				}
				_ => continue,
			};
			conditions.push(active.and(&[&condition]));
		}

		let conditions: Vec<&Ast> = conditions.iter().collect();
		let any_condition = self.ctx.from_bool(false).or(&conditions);
		any_condition._eq(&self.trapped(&pc_next))
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

	pub fn stack_type(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_type_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "stack_type")),
			&[
				&self.ctx.int_sort(), // instruction counter
				&self.ctx.int_sort(), // stack address
			],
			&value_type_sort(self.ctx).0,
		);

		stack_type_func.apply(&[pc, index])
	}

	pub fn program(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let program_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "program")),
			&[&self.ctx.int_sort()],
			&instruction_datatype(&self.ctx).sort,
		);

		program_func.apply(&[pc])
	}

	pub fn local(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let local_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "local")),
			&[&self.ctx.int_sort(), &self.ctx.int_sort()],
			&self.ctx.bitvector_sort(32),
		);

		local_func.apply(&[pc, index])
	}

	pub fn trapped(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let trapped_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "trapped")),
			&[&self.ctx.int_sort()],
			&self.ctx.bool_sort(),
		);

		trapped_func.apply(&[pc])
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
	use parity_wasm::elements::ValueType::*;
	use Instruction::*;

	#[test]
	fn source_program() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[I32Const(1), Nop, I32Const(2), I32Add];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		for (i, instr) in program.iter().enumerate() {
			let instr_enc = state.program(&ctx.from_usize(i));
			let is_equal = instruction_datatype(&ctx).variants[instr.as_usize()]
				.tester
				.apply(&[&instr_enc]);
			let b = model.eval(&is_equal).unwrap().as_bool().unwrap();
			assert!(b);
		}
	}

	#[test]
	fn initial_conditions() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[I32Const(1), I32Const(2), I32Add];

		let constants = Constants::new(&ctx, vec![], &[]);
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

		let program = &[I32Const(1), I32Const(2), I32Add];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		solver.assert(&state.transitions());

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
	fn program_encode_decode() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[I32Const(1), I32Const(2)];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(state.decode_program(&model), program);
	}

	#[test]
	fn transition_const() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[I32Const(1)];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		solver.assert(&state.transitions());

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

		let program = &[I32Const(1), Nop, I32Const(2), I32Add];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		solver.assert(&state.transitions());

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

		let program = &[I32Add];

		let constants = Constants::new(&ctx, vec![], &[I32, I32]);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		solver.assert(&state.transitions());

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

		let program = &[I32Const(1), I32Drop];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		solver.assert(&state.transitions());

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

		let program = &[I32Const(1), I32Const(2), I32Const(3), I32Select];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		solver.assert(&state.transitions());

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

		let program = &[I32Const(1), I32Const(2), I32Const(0), I32Select];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		solver.assert(&state.transitions());

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
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			// x2 = x1 + x0
			I32GetLocal(0),
			I32GetLocal(1),
			I32Add,
			I32SetLocal(2),
			// swap x0, x1
			// tmp = x1; x1 = x0; x0 = tmp;
			I32GetLocal(1),
			I32GetLocal(0),
			I32SetLocal(1),
			I32TeeLocal(0),
		];

		let initial_locals = vec![
			ctx.from_usize(1).int2bv(32),
			ctx.from_usize(2).int2bv(32),
			ctx.from_usize(0).int2bv(32),
		];

		let constants = Constants::new(&ctx, initial_locals, &[]);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);

		solver.assert(&state.n_locals()._eq(&ctx.from_usize(3)));

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

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

		// locals keep their values if not changed
		assert_eq!(local(1, 0), 1);
		assert_eq!(local(1, 1), 2);
		assert_eq!(local(1, 2), 0);

		assert_eq!(stack_pointer(1), 1);
		assert_eq!(stack(1, 0), 1);
		assert_eq!(stack_pointer(2), 2);
		assert_eq!(stack(2, 1), 2);

		// correct value before set_local
		assert_eq!(stack_pointer(3), 1);
		assert_eq!(stack(3, 0), 3);

		assert_eq!(stack_pointer(4), 0);
		assert_eq!(local(4, 2), 3);

		assert_eq!(stack_pointer(8), 1);
		assert_eq!(stack(8, 0), 2);
		assert_eq!(local(8, 0), 2);
		assert_eq!(local(8, 1), 1);
	}

	#[test]
	fn transition_trapped_unreachable() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Unreachable, Nop];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let trapped = |i| {
			model
				.eval(&state.trapped(&ctx.from_usize(i)))
				.unwrap()
				.as_bool()
				.unwrap()
		};

		assert!(!trapped(0));
		assert!(trapped(1));
		assert!(trapped(2));
	}

	#[test]
	fn transition_trapped_div0() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[I32Const(0), I32Const(1), I32DivU];

		let constants = Constants::new(&ctx, vec![], &[]);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let trapped = |i| {
			model
				.eval(&state.trapped(&ctx.from_usize(i)))
				.unwrap()
				.as_bool()
				.unwrap()
		};

		assert!(!trapped(0));
		assert!(!trapped(1));
		assert!(!trapped(2));
		assert!(trapped(3));
	}
}
