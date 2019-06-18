use parity_wasm::elements::Instruction;

pub static IMPLEMENTED_INSTRUCTIONS: &'static [(Instruction, &'static str)] = &[
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

pub fn stack_pop_push_count(i: &Instruction) -> (usize, usize) {
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

pub fn iter_instructions() -> impl Iterator<Item = &'static Instruction> {
	IMPLEMENTED_INSTRUCTIONS.iter().map(|i| &i.0)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Constants;
	use z3::*;

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

		let eval = |ast: &Ast| -> usize {
			let ast = model.eval(ast).unwrap();
			ast.as_usize().unwrap()
		};

		for i in iter_instructions() {
			let (pops, pushs) = stack_pop_push_count(i);
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

}
