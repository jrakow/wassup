use crate::instructions::{instruction_datatype, Instruction};
use crate::*;
use z3::*;

pub struct State<'ctx, 'solver, 'constants> {
	pub ctx: &'ctx Context,
	pub solver: &'solver Solver<'ctx>,
	pub constants: &'constants Constants<'ctx>,
	pub prefix: String,
	pub program_length: usize,
}

impl<'ctx, 'solver, 'constants> State<'ctx, 'solver, 'constants> {
	pub fn new(
		ctx: &'ctx Context,
		solver: &'solver Solver<'ctx>,
		constants: &'constants Constants<'ctx>,
		prefix: &str,
		program_length: usize,
	) -> Self {
		let state = State {
			ctx,
			solver,
			constants,
			prefix: prefix.to_string(),
			program_length,
		};

		state.set_initial();

		state
	}

	pub fn set_initial(&self) {
		// TODO
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

		// set local(0, i) = inital_locals[i]
		for (i, var) in self.constants.initial_locals.iter().enumerate() {
			self.solver.assert(
				&self
					.local(&self.ctx.from_usize(0), &self.ctx.from_usize(i))
					._eq(&var),
			);
		}

		// TODO constrain stack_pointer - stack_pops >= 0

		// constrain 0 <= local_index < n_locals
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = in_range(&self.ctx.from_usize(0), &pc, &self.ctx.from_usize(self.program_length));
		let instr = self.program(&pc);
		let mut conditions = Vec::new();
		for i in &[
			Instruction::GetLocal(0),
			Instruction::SetLocal(0),
			Instruction::TeeLocal(0),
		] {
			let variant = &instruction_datatype(self.ctx, self.constants.value_type_config)
				.variants[i.as_usize(self.constants.value_type_config)];
			let active = variant.tester.apply(&[&instr]);
			let index = variant.accessors[0].apply(&[&instr]);

			let index_in_range =
				in_range(&self.ctx.from_usize(0), &index, &self.constants.n_locals);

			conditions.push(active.implies(&index_in_range));
		}
		// as_ref
		let conditions: Vec<_> = conditions.iter().collect();
		let combined = self.ctx.from_bool(true).and(&conditions);
		self.solver
			.assert(&self.ctx.forall_const(&[&pc], &pc_in_range.implies(&combined)));
	}

	pub fn set_source_program(&self, program: &[Instruction]) {
		assert_eq!(program.len(), self.program_length);

		for (pc, instruction) in program.iter().enumerate() {
			let pc = self.ctx.from_usize(pc);

			// set program
			self.solver.assert(
				&self
					.program(&pc)
					._eq(&instruction.encode(self.ctx, self.constants.value_type_config)),
			);
		}
	}

	pub fn decode_program(&self, model: &Model) -> Vec<Instruction> {
		let mut program = Vec::new();

		for pc in 0..self.program_length {
			let pc = self.ctx.from_usize(pc);
			let encoded_instr = model.eval(&self.program(&pc)).unwrap();

			let decoded = Instruction::decode(
				&encoded_instr,
				self.ctx,
				model,
				self.constants.value_type_config,
			);

			program.push(decoded);
		}

		program
	}

	pub fn transitions(&self) -> Ast<'ctx> {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = in_range(&self.ctx.from_usize(0), &pc, &self.ctx.from_usize(self.program_length));

		// forall initial_stack values and all params and all pcs
		let mut bounds: Vec<_> = self.constants.initial_stack_bounds.iter().collect();
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

		let instruction_datatype = instruction_datatype(self.ctx, self.constants.value_type_config);
		let instr = self.program(&pc);

		let to_i32 = |i: &Ast<'ctx>| {
			self.constants
				.value_type_config
				.i32_wrap_as_i64(self.ctx, i)
		};
		let as_i32 = |i: &Ast<'ctx>| self.constants.value_type_config.i64_unwrap_as_i32(i);

		// helpers
		let i32_size = self.constants.value_type_config.i32_size;
		let bool_to_i32 = |b: &Ast<'ctx>| {
			to_i32(
				&b.ite(&self.ctx.from_usize(1), &self.ctx.from_usize(0))
					.int2bv(i32_size as u64),
			)
		};
		let bvmod32 =
			|b: &Ast<'ctx>| b.bvurem(&self.ctx.from_usize(i32_size).int2bv(i32_size as u64));
		let bvmod64 = |b: &Ast<'ctx>| {
			let i64_size = self.constants.value_type_config.i64_size.unwrap();
			b.bvurem(&self.ctx.from_usize(i64_size).int2bv(i64_size as u64))
		};
		let bv64_zero = || {
			let i64_size = self.constants.value_type_config.i64_size.unwrap();
			self.ctx.from_usize(0).int2bv(i64_size as u64)
		};

		let bv32_zero = self.ctx.from_usize(0).int2bv(i32_size as u64);
		let pc_next = &pc.add(&[&self.ctx.from_usize(1)]);

		let op = |i| self.stack(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(i)]));
		let op_type =
			|i| self.stack_type(&pc, &self.stack_pointer(&pc).sub(&[&self.ctx.from_i64(i)]));
		let result = self.stack(
			pc_next,
			&self.stack_pointer(&pc_next).sub(&[&self.ctx.from_i64(1)]),
		);
		let result_type = self.stack_type(
			pc_next,
			&self.stack_pointer(&pc_next).sub(&[&self.ctx.from_i64(1)]),
		);
		let i32_type = self
			.constants
			.value_type_config
			.encode_value_type(self.ctx, ValueType::I32);
		let i64_type = || {
			self.constants
				.value_type_config
				.encode_value_type(self.ctx, ValueType::I64)
		};

		let mut transitions = Vec::new();
		for (i, variant) in instruction_datatype.variants.iter().enumerate() {
			let active = variant.tester.apply(&[&instr]);
			let template = Instruction::iter_templates(self.constants.value_type_config)
				.nth(i)
				.unwrap();

			let operand_types_correct = match template {
				I32Eqz => op_type(1)._eq(&i32_type),
				// irelop
				I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
				// ibinop
				I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr
				=> op_type(1)._eq(&i32_type).and(&[&op_type(2)._eq(&i32_type)]),

				I64Eqz => op_type(1)._eq(&i64_type()),
				// irelop
				I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
				// ibinop
				I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr
				=> op_type(1)._eq(&i64_type()).and(&[&op_type(2)._eq(&i64_type())]),

				Select => op_type(1)._eq(&i32_type).and(&[&op_type(2)._eq(&op_type(3))]),
				SetLocal(_) | TeeLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);
					let ty = self.constants.local_type(&index);
					op_type(1)._eq(&ty)
				}

				_ => self.ctx.from_bool(true),
			};

			let result_type_correct = match template {
				I32Eqz | I64Eqz |
				// irelop 32
				I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
				// irelop 64
				I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
				// ibinop 32
				I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr
				=> result_type._eq(&i32_type),

				// ibinop
				I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr
				=> result_type._eq(&i64_type()),

				Select => result_type._eq(&op_type(2)),
				Const(_) => result_type._eq(&variant.accessors[1].apply(&[&instr])),
				GetLocal(_) | TeeLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);
					let ty = self.constants.local_type(&index);
					result_type._eq(&ty)
				}

				_ => self.ctx.from_bool(true),
			};

			let next_trapped = match template {
				Unreachable => self.ctx.from_bool(true),

				I32DivU | I32RemU | I32RemS => as_i32(&op(2))._eq(&bv32_zero),
				I32DivS => {
					let divide_by_zero = as_i32(&op(2))._eq(&bv32_zero);
					let overflow = as_i32(&op(2)).bvsdiv_no_overflow(&as_i32(&op(1)));
					divide_by_zero.or(&[&overflow])
				}
				I64DivU | I64RemU | I64RemS => op(2)._eq(&bv64_zero()),
				I64DivS => {
					let divide_by_zero = op(2)._eq(&bv64_zero());
					let overflow = &op(2).bvsdiv_no_overflow(&op(1));
					divide_by_zero.or(&[&overflow])
				}
				_ => self.ctx.from_bool(false),
			};
			let next_trapped = self.trapped(&pc).or(&[&next_trapped]);
			let trapped_transition = self.trapped(pc_next)._eq(&next_trapped);

			let transition = match template {
				Unreachable | Nop | Drop => self.ctx.from_bool(true),

				Const(_) => result._eq(&variant.accessors[0].apply(&[&instr])),

				I32Eqz => result._eq(&bool_to_i32(&as_i32(&op(1))._eq(&bv32_zero))),
				I32Eq => result._eq(&bool_to_i32(&as_i32(&op(2))._eq(&as_i32(&op(1))))),
				I32Ne => result._eq(&bool_to_i32(&as_i32(&op(2))._eq(&as_i32(&op(1))).not())),
				I32LtS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvslt(&as_i32(&op(1))))),
				I32LtU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvult(&as_i32(&op(1))))),
				I32GtS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvsgt(&as_i32(&op(1))))),
				I32GtU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvugt(&as_i32(&op(1))))),
				I32LeS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvsle(&as_i32(&op(1))))),
				I32LeU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvule(&as_i32(&op(1))))),
				I32GeS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvsge(&as_i32(&op(1))))),
				I32GeU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvuge(&as_i32(&op(1))))),

				I64Eqz => result._eq(&bool_to_i32(&op(1)._eq(&bv64_zero()))),
				I64Eq => result._eq(&bool_to_i32(&op(2)._eq(&op(1)))),
				I64Ne => result._eq(&bool_to_i32(&op(2)._eq(&op(1)).not())),
				I64LtS => result._eq(&bool_to_i32(&op(2).bvslt(&op(1)))),
				I64LtU => result._eq(&bool_to_i32(&op(2).bvult(&op(1)))),
				I64GtS => result._eq(&bool_to_i32(&op(2).bvsgt(&op(1)))),
				I64GtU => result._eq(&bool_to_i32(&op(2).bvugt(&op(1)))),
				I64LeS => result._eq(&bool_to_i32(&op(2).bvsle(&op(1)))),
				I64LeU => result._eq(&bool_to_i32(&op(2).bvule(&op(1)))),
				I64GeS => result._eq(&bool_to_i32(&op(2).bvsge(&op(1)))),
				I64GeU => result._eq(&bool_to_i32(&op(2).bvuge(&op(1)))),

				I32Add => result._eq(&to_i32(&as_i32(&op(2)).bvadd(&as_i32(&op(1))))),
				I32Sub => result._eq(&to_i32(&as_i32(&op(2)).bvsub(&as_i32(&op(1))))),
				I32Mul => result._eq(&to_i32(&as_i32(&op(2)).bvmul(&as_i32(&op(1))))),
				I32DivS => result._eq(&to_i32(&as_i32(&op(2)).bvsdiv(&as_i32(&op(1))))),
				I32DivU => result._eq(&to_i32(&as_i32(&op(2)).bvudiv(&as_i32(&op(1))))),
				I32RemS => result._eq(&to_i32(&as_i32(&op(2)).bvsrem(&as_i32(&op(1))))),
				I32RemU => result._eq(&to_i32(&as_i32(&op(2)).bvurem(&as_i32(&op(1))))),
				I32And => result._eq(&to_i32(&as_i32(&op(2)).bvand(&as_i32(&op(1))))),
				I32Or => result._eq(&to_i32(&as_i32(&op(2)).bvor(&as_i32(&op(1))))),
				I32Xor => result._eq(&to_i32(&as_i32(&op(2)).bvxor(&as_i32(&op(1))))),
				I32Shl => result._eq(&to_i32(&as_i32(&op(2)).bvshl(&bvmod32(&as_i32(&op(1)))))),
				I32ShrS => result._eq(&to_i32(&as_i32(&op(2)).bvashr(&bvmod32(&as_i32(&op(1)))))),
				I32ShrU => result._eq(&to_i32(&as_i32(&op(2)).bvlshr(&bvmod32(&as_i32(&op(1)))))),
				I32Rotl => result._eq(&to_i32(&as_i32(&op(2)).bvrotl(&bvmod32(&as_i32(&op(1)))))),
				I32Rotr => result._eq(&to_i32(&as_i32(&op(2)).bvrotr(&bvmod32(&as_i32(&op(1)))))),

				I64Add => result._eq(&op(2).bvadd(&op(1))),
				I64Sub => result._eq(&op(2).bvsub(&op(1))),
				I64Mul => result._eq(&op(2).bvmul(&op(1))),
				I64DivS => result._eq(&op(2).bvsdiv(&op(1))),
				I64DivU => result._eq(&op(2).bvudiv(&op(1))),
				I64RemS => result._eq(&op(2).bvsrem(&op(1))),
				I64RemU => result._eq(&op(2).bvurem(&op(1))),
				I64And => result._eq(&op(2).bvand(&op(1))),
				I64Or => result._eq(&op(2).bvor(&op(1))),
				I64Xor => result._eq(&op(2).bvxor(&op(1))),
				I64Shl => result._eq(&op(2).bvshl(&bvmod64(&op(1)))),
				I64ShrS => result._eq(&op(2).bvashr(&bvmod64(&op(1)))),
				I64ShrU => result._eq(&op(2).bvlshr(&bvmod64(&op(1)))),
				I64Rotl => result._eq(&op(2).bvrotl(&bvmod64(&op(1)))),
				I64Rotr => result._eq(&op(2).bvrotr(&bvmod64(&op(1)))),

				Select => result._eq(&as_i32(&op(1))._eq(&bv32_zero).ite(&op(2), &op(3))),

				GetLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);

					result._eq(&self.local(&pc, &index))
				}
				SetLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);

					self.local(&pc_next, &index)._eq(&op(1))
				}
				TeeLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);

					let local_is_set = self.local(&pc_next, &index)._eq(&op(1));
					let stack_is_set = result._eq(&op(1));

					local_is_set.and(&[&stack_is_set])
				}
			};

			transitions.push(active.implies(&transition.and(&[
				&operand_types_correct,
				&result_type_correct,
				&trapped_transition,
			])));
		}

		// create vector of references
		let transitions: Vec<_> = transitions.iter().collect();
		self.ctx.from_bool(true).and(&transitions)
	}

	fn preserve_stack(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		// constants
		let instr = self.program(&pc);
		let stack_pointer = self.stack_pointer(&pc);
		let pc_next = pc.add(&[&self.ctx.from_usize(1)]);

		// preserve stack values stack(_, 0)..=stack(_, stack_pointer - pops - 1)
		let n = self.ctx.named_int_const("n");

		let n_in_range = in_range(
			&self.ctx.from_usize(0),
			&n,
			&stack_pointer.sub(&[&self.constants.stack_pop_count(&instr)]),
		);
		let slot_preserved = self.stack(&pc, &n)._eq(&self.stack(&pc_next, &n));
		let type_preserved = self.stack_type(&pc, &n)._eq(&self.stack_type(&pc_next, &n));

		// forall n
		self.ctx.forall_const(
			&[&n],
			&n_in_range.implies(&slot_preserved.and(&[&type_preserved])),
		)
	}

	fn preserve_locals(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let instr = self.program(&pc);

		// preserve all locals which are not set in this step
		let i = self.ctx.named_int_const("i");
		let i_in_range = in_range(&self.ctx.from_usize(0), &i, &self.constants.n_locals);

		let variants = &instruction_datatype(self.ctx, self.constants.value_type_config).variants;

		// disable if set_local
		let set_local =
			&variants[Instruction::SetLocal(0).as_usize(self.constants.value_type_config)];
		let is_set_local = set_local.tester.apply(&[&instr]);
		let set_local_index = set_local.accessors[0].apply(&[&instr]);
		let set_local_active = is_set_local.and(&[&set_local_index._eq(&i)]);

		let tee_local =
			&variants[Instruction::TeeLocal(0).as_usize(self.constants.value_type_config)];
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
			&self.constants.value_type_config.value_sort(self.ctx),
		);

		stack_func.apply(&[pc, index])
	}

	pub fn stack_type(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_type_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "stack_type")),
			&[
				&self.ctx.int_sort(), // program counter
				&self.ctx.int_sort(), // index
			],
			&self
				.constants
				.value_type_config
				.value_type_datatype(self.ctx)
				.sort,
		);

		stack_type_func.apply(&[pc, index])
	}

	pub fn program(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		let program_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "program")),
			&[&self.ctx.int_sort()],
			&instruction_datatype(&self.ctx, self.constants.value_type_config).sort,
		);

		program_func.apply(&[pc])
	}

	pub fn local(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let local_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "local")),
			&[&self.ctx.int_sort(), &self.ctx.int_sort()],
			&self.constants.value_type_config.value_sort(self.ctx),
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
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;
	use Value::*;

	#[test]
	fn source_program() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Nop, Const(I32(2)), I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 4);

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		for (i, instr) in program.iter().enumerate() {
			let instr_enc = state.program(&ctx.from_usize(i));
			let is_equal = instruction_datatype(&ctx, value_type_config).variants
				[instr.as_usize(value_type_config)]
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

		let program = &[Const(I32(1)), Const(I32(2)), I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 3);

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

		let program = &[Const(I32(1)), Const(I32(2)), I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 3);

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

		let program = &[Const(I32(1)), Const(I32(2))];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 2);

		state.set_source_program(program);

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(state.decode_program(&model), program);
	}

	#[test]
	fn transition_const() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1))];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 1);

		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let eval_int = |ast| -> i64 {
			let evaled = model.eval(ast).unwrap();
			evaled.as_i64().unwrap()
		};
		assert_eq!(eval_int(&state.stack_pointer(&ctx.from_usize(1))), 1);

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		assert_eq!(stack(1, 0), Value::I32(1));
	}

	#[test]
	fn transition_add_consts() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Nop, Const(I32(2)), I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 4);

		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		assert_eq!(stack(1, 0), Value::I32(1));
		assert_eq!(stack(2, 0), Value::I32(1));
		assert_eq!(stack(3, 0), Value::I32(1));
		assert_eq!(stack(3, 1), Value::I32(2));
		assert_eq!(stack(4, 0), Value::I32(3));
	}

	#[test]
	fn transition_add_consts_config_i32() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Nop, Const(I32(2)), I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: None,
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 4);

		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		assert_eq!(stack(1, 0), Value::I32(1));
		assert_eq!(stack(2, 0), Value::I32(1));
		assert_eq!(stack(3, 0), Value::I32(1));
		assert_eq!(stack(3, 1), Value::I32(2));
		assert_eq!(stack(4, 0), Value::I32(3));
	}

	#[test]
	fn transition_add_consts_config_reduced_size() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I64(1)), Nop, Const(I64(2)), I64Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 8,
			i64_size: Some(16),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 4);

		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		assert_eq!(stack(1, 0), Value::I64(1));
		assert_eq!(stack(2, 0), Value::I64(1));
		assert_eq!(stack(3, 0), Value::I64(1));
		assert_eq!(stack(3, 1), Value::I64(2));
		assert_eq!(stack(4, 0), Value::I64(3));
	}

	#[test]
	fn transition_add() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(
			&ctx,
			&solver,
			vec![],
			vec![],
			&[ValueType::I32, ValueType::I32],
			value_type_config,
		);
		let state = State::new(&ctx, &solver, &constants, "", 1);

		state.set_source_program(program);

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		let x0 = value_type_config.i64_unwrap_as_i32(&constants.initial_stack[0]);
		let x1 = value_type_config.i64_unwrap_as_i32(&constants.initial_stack[1]);
		let sum = x0.bvadd(&x1);

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};

		assert_eq!(
			stack(1, 0),
			Value::I32(model.eval(&sum.bv2int(true)).unwrap().as_i32().unwrap())
		);
	}

	#[test]
	fn transition_const_drop() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Drop];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 2);
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

		let program = &[Const(I32(1)), Const(I32(2)), Const(I32(3)), Select];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 4);
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

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		assert_eq!(stack(4, 0), Value::I32(1));
	}

	#[test]
	fn transition_consts_select_false() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Const(I32(2)), Const(I32(0)), Select];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 4);
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

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		assert_eq!(stack(4, 0), Value::I32(2));
	}

	#[test]
	fn transition_local() {
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
			TeeLocal(0),
		];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};

		let initial_locals = vec![
			Value::I32(1).encode(&ctx, value_type_config),
			Value::I32(2).encode(&ctx, value_type_config),
			Value::I32(0).encode(&ctx, value_type_config),
		];

		let constants = Constants::new(
			&ctx,
			&solver,
			initial_locals,
			vec![ValueType::I32; 3],
			&[],
			value_type_config,
		);

		let state = State::new(&ctx, &solver, &constants, "", 8);
		state.set_source_program(program);

		solver.assert(&constants.n_locals._eq(&ctx.from_usize(3)));

		solver.assert(&state.transitions());

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model.eval(&constants.n_locals).unwrap().as_usize().unwrap(),
			3
		);

		let stack_pointer = |pc| {
			model
				.eval(&state.stack_pointer(&ctx.from_usize(pc)))
				.unwrap()
				.as_usize()
				.unwrap()
		};

		let stack = |pc: usize, i: usize| -> Value {
			let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

			Value::decode(
				value,
				&model,
				value_type_config.decode_value_type(&ctx, &model, ty),
				value_type_config,
			)
		};
		let local = |pc: usize, i: usize| -> Value {
			let value = &state.local(&ctx.from_usize(pc), &ctx.from_usize(i));
			let ty = ValueType::I32;

			Value::decode(value, &model, ty, value_type_config)
		};

		assert_eq!(local(0, 0), Value::I32(1));
		assert_eq!(local(0, 1), Value::I32(2));
		// default value
		assert_eq!(local(0, 2), Value::I32(0));

		// locals keep their values if not changed
		assert_eq!(local(1, 0), Value::I32(1));
		assert_eq!(local(1, 1), Value::I32(2));
		assert_eq!(local(1, 2), Value::I32(0));

		assert_eq!(stack_pointer(1), 1);
		assert_eq!(stack(1, 0), Value::I32(1));
		assert_eq!(stack_pointer(2), 2);
		assert_eq!(stack(2, 1), Value::I32(2));

		// correct value before set_local
		assert_eq!(stack_pointer(3), 1);
		assert_eq!(stack(3, 0), Value::I32(3));

		assert_eq!(stack_pointer(4), 0);
		assert_eq!(local(4, 2), Value::I32(3));

		assert_eq!(stack_pointer(8), 1);
		assert_eq!(stack(8, 0), Value::I32(2));
		assert_eq!(local(8, 0), Value::I32(2));
		assert_eq!(local(8, 1), Value::I32(1));
	}

	#[test]
	fn transition_trapped_unreachable() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Unreachable, Nop];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 2);
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

		let program = &[Const(I32(0)), Const(I32(1)), I32DivU];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
		let state = State::new(&ctx, &solver, &constants, "", 3);
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
