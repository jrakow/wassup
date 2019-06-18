use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use z3::*;

static IMPLEMENTED_INSTRUCTIONS: &'static [(Instruction, &'static str)] = &[
	//	Unreachable,
	(Instruction::Nop, "Nop"),
	//	Block(BlockType),
	//	Loop(BlockType),
	//	If(BlockType),
	//	Else,
	//	End,
	//	Br(u32),
	//	BrIf(u32),
	//	BrTable(Box<BrTableData>),
	//	Return,

	//	Call(u32),
	//	CallIndirect(u32, u8),
	(Instruction::Drop, "Drop"),
	(Instruction::Select, "Select"),
	(Instruction::GetLocal(0), "GetLocal"),
	(Instruction::SetLocal(0), "SetLocal"),
	(Instruction::TeeLocal(0), "TeeLocal"),
	//	GetGlobal(u32),
	//	SetGlobal(u32),

	//	I32Load(u32, u32),
	//	I64Load(u32, u32),
	//	F32Load(u32, u32),
	//	F64Load(u32, u32),
	//	I32Load8S(u32, u32),
	//	I32Load8U(u32, u32),
	//	I32Load16S(u32, u32),
	//	I32Load16U(u32, u32),
	//	I64Load8S(u32, u32),
	//	I64Load8U(u32, u32),
	//	I64Load16S(u32, u32),
	//	I64Load16U(u32, u32),
	//	I64Load32S(u32, u32),
	//	I64Load32U(u32, u32),
	//	I32Store(u32, u32),
	//	I64Store(u32, u32),
	//	F32Store(u32, u32),
	//	F64Store(u32, u32),
	//	I32Store8(u32, u32),
	//	I32Store16(u32, u32),
	//	I64Store8(u32, u32),
	//	I64Store16(u32, u32),
	//	I64Store32(u32, u32),

	//	CurrentMemory(u8),
	//	GrowMemory(u8),

	// numeric instructions
	(Instruction::I32Const(0), "I32Const"),
	//	(I64Const(0), "I64Const"),
	//	(F32Const(0), "F32Const"),
	//	(F64Const(0), "F64Const"),
	(Instruction::I32Eqz, "I32Eqz"),
	(Instruction::I32Eq, "I32Eq"),
	(Instruction::I32Ne, "I32Ne"),
	(Instruction::I32LtS, "I32LtS"),
	(Instruction::I32LtU, "I32LtU"),
	(Instruction::I32GtS, "I32GtS"),
	(Instruction::I32GtU, "I32GtU"),
	(Instruction::I32LeS, "I32LeS"),
	(Instruction::I32LeU, "I32LeU"),
	(Instruction::I32GeS, "I32GeS"),
	(Instruction::I32GeU, "I32GeU"),
	//	(I64Eqz, "I64Eqz"),
	//	(I64Eq, "I64Eq"),
	//	(I64Ne, "I64Ne"),
	//	(I64LtS, "I64LtS"),
	//	(I64LtU, "I64LtU"),
	//	(I64GtS, "I64GtS"),
	//	(I64GtU, "I64GtU"),
	//	(I64LeS, "I64LeS"),
	//	(I64LeU, "I64LeU"),
	//	(I64GeS, "I64GeS"),
	//	(I64GeU, "I64GeU"),

	//	(F32Eq, "F32Eq"),
	//	(F32Ne, "F32Ne"),
	//	(F32Lt, "F32Lt"),
	//	(F32Gt, "F32Gt"),
	//	(F32Le, "F32Le"),
	//	(F32Ge, "F32Ge"),

	//	(F64Eq, "F64Eq"),
	//	(F64Ne, "F64Ne"),
	//	(F64Lt, "F64Lt"),
	//	(F64Gt, "F64Gt"),
	//	(F64Le, "F64Le"),
	//	(F64Ge, "F64Ge"),
	//	(I32Clz, "I32Clz"),
	//	(I32Ctz, "I32Ctz"),
	//	(I32Popcnt, "I32Popcnt"),
	(Instruction::I32Add, "I32Add"),
	(Instruction::I32Sub, "I32Sub"),
	(Instruction::I32Mul, "I32Mul"),
	(Instruction::I32DivS, "I32DivS"),
	(Instruction::I32DivU, "I32DivU"),
	(Instruction::I32RemS, "I32RemS"),
	(Instruction::I32RemU, "I32RemU"),
	(Instruction::I32And, "I32And"),
	(Instruction::I32Or, "I32Or"),
	(Instruction::I32Xor, "I32Xor"),
	(Instruction::I32Shl, "I32Shl"),
	(Instruction::I32ShrS, "I32ShrS"),
	(Instruction::I32ShrU, "I32ShrU"),
	(Instruction::I32Rotl, "I32Rotl"),
	(Instruction::I32Rotr, "I32Rotr"),
	//	(I64Clz, "I64Clz"),
	//	(I64Ctz, "I64Ctz"),
	//	(I64Popcnt, "I64Popcnt"),
	//	(I64Add, "I64Add"),
	//	(I64Sub, "I64Sub"),
	//	(I64Mul, "I64Mul"),
	//	(I64DivS, "I64DivS"),
	//	(I64DivU, "I64DivU"),
	//	(I64RemS, "I64RemS"),
	//	(I64RemU, "I64RemU"),
	//	(I64And, "I64And"),
	//	(I64Or, "I64Or"),
	//	(I64Xor, "I64Xor"),
	//	(I64Shl, "I64Shl"),
	//	(I64ShrS, "I64ShrS"),
	//	(I64ShrU, "I64ShrU"),
	//	(I64Rotl, "I64Rotl"),
	//	(I64Rotr, "I64Rotr"),

	//	(F32Abs, "F32Abs"),
	//	(F32Neg, "F32Neg"),
	//	(F32Ceil, "F32Ceil"),
	//	(F32Floor, "F32Floor"),
	//	(F32Trunc, "F32Trunc"),
	//	(F32Nearest, "F32Nearest"),
	//	(F32Sqrt, "F32Sqrt"),
	//	(F32Add, "F32Add"),
	//	(F32Sub, "F32Sub"),
	//	(F32Mul, "F32Mul"),
	//	(F32Div, "F32Div"),
	//	(F32Min, "F32Min"),
	//	(F32Max, "F32Max"),
	//	(F32Copysign, "F32Copysign"),
	//	(F64Abs, "F64Abs"),
	//	(F64Neg, "F64Neg"),
	//	(F64Ceil, "F64Ceil"),
	//	(F64Floor, "F64Floor"),
	//	(F64Trunc, "F64Trunc"),
	//	(F64Nearest, "F64Nearest"),
	//	(F64Sqrt, "F64Sqrt"),
	//	(F64Add, "F64Add"),
	//	(F64Sub, "F64Sub"),
	//	(F64Mul, "F64Mul"),
	//	(F64Div, "F64Div"),
	//	(F64Min, "F64Min"),
	//	(F64Max, "F64Max"),
	//	(F64Copysign, "F64Copysign"),

	//	(I32WrapI64, "I32WrapI64"),
	//	(I32TruncSF32, "I32TruncSF32"),
	//	(I32TruncUF32, "I32TruncUF32"),
	//	(I32TruncSF64, "I32TruncSF64"),
	//	(I32TruncUF64, "I32TruncUF64"),
	//	(I64ExtendSI32, "I64ExtendSI32"),
	//	(I64ExtendUI32, "I64ExtendUI32"),
	//	(I64TruncSF32, "I64TruncSF32"),
	//	(I64TruncUF32, "I64TruncUF32"),
	//	(I64TruncSF64, "I64TruncSF64"),
	//	(I64TruncUF64, "I64TruncUF64"),

	//	(F32ConvertSI32, "F32ConvertSI32"),
	//	(F32ConvertUI32, "F32ConvertUI32"),
	//	(F32ConvertSI64, "F32ConvertSI64"),
	//	(F32ConvertUI64, "F32ConvertUI64"),
	//	(F32DemoteF64, "F32DemoteF64"),
	//	(F64ConvertSI32, "F64ConvertSI32"),
	//	(F64ConvertUI32, "F64ConvertUI32"),
	//	(F64ConvertSI64, "F64ConvertSI64"),
	//	(F64ConvertUI64, "F64ConvertUI64"),
	//	(F64PromoteF32, "F64PromoteF32"),

	//	(I32ReinterpretF32, "I32ReinterpretF32"),
	//	(I64ReinterpretF64, "I64ReinterpretF64"),
	//	(F32ReinterpretI32, "F32ReinterpretI32"),
	//	(F64ReinterpretI64, "F64ReinterpretI64"),

	//	(I32Extend8S, "I32Extend8S"),
	//	(I32Extend16S, "I32Extend16S"),
	//	(I64Extend8S, "I64Extend8S"),
	//	(I64Extend16S, "I64Extend16S"),
	//	(I64Extend32S, "I64Extend32S"),
];

fn instruction_templates_equal(i: &Instruction, j: &Instruction) -> bool {
	use Instruction::*;

	match (i, j) {
		(I32Const(_), I32Const(_)) => true,
		(I64Const(_), I64Const(_)) => true,
		(GetLocal(_), GetLocal(_)) => true,
		(SetLocal(_), SetLocal(_)) => true,
		(TeeLocal(_), TeeLocal(_)) => true,
		_ => i == j,
	}
}

pub fn instruction_to_index(i: &Instruction) -> usize {
	IMPLEMENTED_INSTRUCTIONS
		.iter()
		.position(|j| instruction_templates_equal(i, &j.0))
		.unwrap_or_else(|| unimplemented!())
}

fn stack_pop_push_count(i: &Instruction) -> (u64, u64) {
	use Instruction::*;

	match i {
		Nop => (0, 0),

		I32Const(_) => (0, 1),

		// itestop
		I32Eqz => (1, 1),
		// irelop
		I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU => {
			(2, 1)
		}
		// iunop
		I32Clz | I32Ctz | I32Popcnt => (1, 1),
		// ibinop
		I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or
		| I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr => (2, 1),

		// parametric
		Drop => (1, 0),
		Select => (3, 1),

		// locals
		GetLocal(_) => (0, 1),
		SetLocal(_) => (1, 0),
		TeeLocal(_) => (0, 0),

		_ => unimplemented!(),
	}
}

pub fn iter_intructions() -> impl Iterator<Item = &'static Instruction> {
	IMPLEMENTED_INSTRUCTIONS.iter().map(|i| &i.0)
}

pub fn stack_depth(program: &[Instruction]) -> u64 {
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

pub fn in_range<'ctx>(a: &Ast<'ctx>, b: &Ast<'ctx>, c: &Ast<'ctx>) -> Ast<'ctx> {
	a.le(&b).and(&[&b.lt(&c)])
}

pub struct Constants<'ctx> {
	pub ctx: &'ctx Context,
	pub word_sort: Sort<'ctx>,
	pub instruction_sort: Sort<'ctx>,
	pub instruction_consts: Vec<FuncDecl<'ctx>>,
	pub instruction_testers: Vec<FuncDecl<'ctx>>,
	pub initial_stack: Vec<Ast<'ctx>>,
	pub params: Vec<Ast<'ctx>>,
	pub stack_depth: usize,
}

impl<'ctx, 'solver> Constants<'ctx> {
	pub fn new(
		ctx: &'ctx Context,
		solver: &Solver<'ctx>,
		stack_depth: usize,
		n_params: usize,
	) -> Self {
		let word_sort = ctx.bitvector_sort(32);
		let instruction_names: Vec<_> = IMPLEMENTED_INSTRUCTIONS
			.iter()
			.map(|i| i.1)
			.map(|s| ctx.str_sym(s))
			.collect();
		let (instruction_sort, instruction_consts, instruction_testers) = ctx.enumeration_sort(
			&ctx.str_sym("Instruction"),
			&instruction_names.iter().collect::<Vec<_>>()[..],
		);
		let initial_stack: Vec<_> = (0..stack_depth)
			.map(|_| ctx.fresh_const("initial_stack", &word_sort))
			.collect();
		let params: Vec<_> = (0..n_params)
			.map(|_| ctx.fresh_const("param", &word_sort))
			.collect();

		let constants = Constants {
			ctx,
			word_sort,
			instruction_sort,
			instruction_consts,
			instruction_testers,
			initial_stack,
			params,
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

		constants
	}

	pub fn instruction(&self, i: &Instruction) -> Ast<'ctx> {
		self.instruction_consts[instruction_to_index(i)].apply(&[])
	}

	fn stack_pop_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_pop_count_func = self.ctx.func_decl(
			self.ctx.str_sym("stack_pop_count"),
			&[&self.instruction_sort],
			&self.ctx.int_sort(),
		);

		stack_pop_count_func.apply(&[instr])
	}

	fn stack_push_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let stack_push_count_func = self.ctx.func_decl(
			self.ctx.str_sym("stack_push_count"),
			&[&self.instruction_sort],
			&self.ctx.int_sort(),
		);

		stack_push_count_func.apply(&[instr])
	}

	pub fn set_params(&self, solver: &Solver, params: &[u32]) {
		for (i, v) in params.iter().enumerate() {
			let v = self.ctx.from_u64(*v as _).int2bv(32);

			solver.assert(&self.params[i]._eq(&v));
		}
	}

	fn int2word(&self, i: &Ast<'ctx>) -> Ast<'ctx> {
		i.int2bv(32)
	}
}

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

		// set params
		for (i, var) in self.constants.params.iter().enumerate() {
			self.solver.assert(
				&self
					.local(&self.ctx.from_u64(0), &self.ctx.from_u64(i as _))
					._eq(&var),
			);
		}

		// force n_locals to be >= n_params
		let n_params = self.ctx.from_u64(self.constants.params.len() as _);
		self.solver.assert(&self.n_locals().ge(&n_params));

		// set remaining locals to 0
		let n = self.ctx.named_int_const("n");
		let bv_zero = self.constants.int2word(&self.ctx.from_u64(0));
		let n_in_range = in_range(&n_params, &n, &self.n_locals());
		self.solver.assert(&self.ctx.forall_const(
			&[&n],
			&n_in_range.implies(&self.local(&self.ctx.from_u64(0), &n)._eq(&bv_zero)),
		));

		// constrain 0 <= local_index <= n_locals
		let pc = self.ctx.named_int_const("pc");
		let local_index_in_range = in_range(
			&self.ctx.from_u64(0),
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
				.assert(&self.program(&self.ctx.from_u64(pc as _))._eq(&instruction))
		}

		for (pc, instr) in program.iter().enumerate() {
			use Instruction::*;
			let pc = self.ctx.from_u64(pc as _);

			// set push_constants function
			match instr {
				I32Const(i) => {
					let i = self
						.constants
						.int2word(&self.ctx.from_i64((*i).try_into().unwrap()));
					self.solver.assert(&self.push_constants(&pc)._eq(&i));
				}
				GetLocal(i) | SetLocal(i) | TeeLocal(i) => {
					self.solver
						.assert(&self.local_index(&pc)._eq(&self.ctx.from_u64(*i as _)));
				}
				_ => {}
			}
		}

		// set length
		self.solver.assert(
			&self
				.program_length()
				._eq(&self.ctx.from_u64(program.len() as u64)),
		);
	}

	pub fn assert_transitions(&self) {
		let pc = self.ctx.named_int_const("pc");
		let pc_in_range = in_range(&self.ctx.from_u64(0), &pc, &self.program_length());

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
		let stack_pointer_next = self.stack_pointer(&pc.add(&[&self.ctx.from_u64(1)]));

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
				&self.ctx.from_u64(1).int2bv(32),
				&self.ctx.from_u64(0).int2bv(32),
			)
		};
		let mod_n = |b: &Ast<'ctx>, n: u64| b.bvurem(&self.ctx.from_u64(n).int2bv(32));

		// constants
		let bv_zero = self.ctx.from_u64(0).int2bv(32);
		let pc_next = &pc.add(&[&self.ctx.from_u64(1)]);

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
			&self.ctx.from_u64(0),
			&n,
			&stack_pointer.sub(&[&self.constants.stack_pop_count(&instr)]),
		);
		let slot_preserved = self
			.stack(&pc, &n)
			._eq(&self.stack(&pc.add(&[&self.ctx.from_u64(1)]), &n));

		// forall n
		self.ctx
			.forall_const(&[&n], &n_in_range.implies(&slot_preserved))
	}

	fn preserve_locals(&self, pc: &Ast<'ctx>) -> Ast<'ctx> {
		// preserve all locals which are not set in this step
		let i = self.ctx.named_int_const("i");
		let i_in_range = in_range(&self.ctx.from_u64(0), &i, &self.n_locals());

		let is_setlocal = self.constants.instruction_testers
			[instruction_to_index(&Instruction::SetLocal(0))]
		.apply(&[&self.program(&pc)]);
		let is_teelocal = self.constants.instruction_testers
			[instruction_to_index(&Instruction::TeeLocal(0))]
		.apply(&[&self.program(&pc)]);
		let is_setting_instruction = is_setlocal.or(&[&is_teelocal]);
		let index_active = i._eq(&self.local_index(&pc));
		let enable = is_setting_instruction.and(&[&index_active]).not();

		let pc_next = pc.add(&[&self.ctx.from_u64(1)]);

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
			&self.constants.word_sort,
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
			&self.constants.word_sort,
		);

		push_constants_func.apply(&[pc])
	}

	pub fn local(&self, pc: &Ast<'ctx>, index: &Ast<'ctx>) -> Ast<'ctx> {
		let local_func = self.ctx.func_decl(
			self.ctx.str_sym(&(self.prefix.to_owned() + "local")),
			&[&self.ctx.int_sort(), &self.ctx.int_sort()],
			&self.constants.word_sort,
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

// this cannot be tested directly as Z3 does not support evaluating terms with quantifiers
pub fn equivalent<'ctx>(
	lhs: &State<'ctx, '_, '_>,
	lhs_pc: &Ast<'ctx>,
	rhs: &State<'ctx, '_, '_>,
	rhs_pc: &Ast<'ctx>,
) -> Ast<'ctx> {
	let ctx = lhs.ctx;

	let stack_pointers_equal = lhs.stack_pointer(&lhs_pc)._eq(&rhs.stack_pointer(&rhs_pc));

	let stacks_equal = {
		// for 0 <= n < stack_pointer
		let n = ctx.named_int_const("n");
		let n_in_range = in_range(&ctx.from_u64(0), &n, &lhs.stack_pointer(&lhs_pc));

		// lhs-stack(lhs_pc, n) ==  rhs-stack(rhs_pc, n)
		let condition = lhs.stack(&lhs_pc, &n)._eq(&rhs.stack(&rhs_pc, &n));

		ctx.forall_const(&[&n], &n_in_range.implies(&condition))
	};

	ctx.from_bool(true)
		.and(&[&stack_pointers_equal, &stacks_equal])
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
		let constants = Constants::new(&ctx, &solver, 0, 0);

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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
		let state = State::new(&ctx, &solver, &constants, "");

		state.set_source_program(program);

		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
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
			eval_bv(&state.stack(&ctx.from_u64(1), &ctx.from_u64(0))),
			eval_bv(&sum)
		);
	}

	#[test]
	fn transition_const_drop() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Instruction::I32Const(1), Instruction::Drop];

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model
				.eval(&state.stack_pointer(&ctx.from_u64(2)))
				.unwrap()
				.as_u64()
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model
				.eval(&state.stack_pointer(&ctx.from_u64(4)))
				.unwrap()
				.as_u64()
				.unwrap(),
			1
		);
		assert_eq!(
			model
				.eval(
					&state
						.stack(&ctx.from_u64(4), &ctx.from_u64(0))
						.bv2int(false)
				)
				.unwrap()
				.as_u64()
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 0);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			model
				.eval(&state.stack_pointer(&ctx.from_u64(4)))
				.unwrap()
				.as_u64()
				.unwrap(),
			1
		);
		assert_eq!(
			model
				.eval(
					&state
						.stack(&ctx.from_u64(4), &ctx.from_u64(0))
						.bv2int(false)
				)
				.unwrap()
				.as_u64()
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

		let constants = Constants::new(&ctx, &solver, stack_depth(program) as _, 2);
		let state = State::new(&ctx, &solver, &constants, "");
		state.set_source_program(program);
		state.assert_transitions();
		constants.set_params(&solver, &[1, 2]);

		println!("{}", &solver);

		assert!(solver.check());
		let model = solver.get_model();

		println!("{}", &model);

		assert_eq!(model.eval(&state.n_locals()).unwrap().as_u64().unwrap(), 3);

		let stack_pointer = |pc| {
			model
				.eval(&state.stack_pointer(&ctx.from_u64(pc)))
				.unwrap()
				.as_u64()
				.unwrap()
		};
		let stack = |pc, i| {
			model
				.eval(
					&state
						.stack(&ctx.from_u64(pc), &ctx.from_u64(i))
						.bv2int(false),
				)
				.unwrap()
				.as_u64()
				.unwrap()
		};
		let local = |pc, i| {
			model
				.eval(
					&state
						.local(&ctx.from_u64(pc), &ctx.from_u64(i))
						.bv2int(false),
				)
				.unwrap()
				.as_u64()
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
