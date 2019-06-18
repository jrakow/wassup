use enum_iterator::IntoEnumIterator;
use parity_wasm::elements::{
	Instruction as PInstruction,
	ValueType::{self, *},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, IntoEnumIterator)]
pub enum Instruction {
	// Unreachable,
	Nop,
	// Block(BlockType),
	// Loop(BlockType),
	// If(BlockType),
	// Else,
	// End,
	// Br(u32),
	// BrIf(u32),
	// BrTable(Box<BrTableData>),
	// Return,

	// Call(u32),
	// CallIndirect(u32, u8),
	I32Drop,
	I32Select,

	I32GetLocal,
	I32SetLocal,
	I32TeeLocal,
	// GetGlobal(u32),
	// SetGlobal(u32),

	// I32Load(u32, u32),
	// I64Load(u32, u32),
	// F32Load(u32, u32),
	// F64Load(u32, u32),
	// I32Load8S(u32, u32),
	// I32Load8U(u32, u32),
	// I32Load16S(u32, u32),
	// I32Load16U(u32, u32),
	// I64Load8S(u32, u32),
	// I64Load8U(u32, u32),
	// I64Load16S(u32, u32),
	// I64Load16U(u32, u32),
	// I64Load32S(u32, u32),
	// I64Load32U(u32, u32),
	// I32Store(u32, u32),
	// I64Store(u32, u32),
	// F32Store(u32, u32),
	// F64Store(u32, u32),
	// I32Store8(u32, u32),
	// I32Store16(u32, u32),
	// I64Store8(u32, u32),
	// I64Store16(u32, u32),
	// I64Store32(u32, u32),

	// CurrentMemory(u8),
	// GrowMemory(u8),
	I32Const,
	// I64Const(i64),
	// F32Const(u32),
	// F64Const(u64),
	I32Eqz,
	I32Eq,
	I32Ne,
	I32LtS,
	I32LtU,
	I32GtS,
	I32GtU,
	I32LeS,
	I32LeU,
	I32GeS,
	I32GeU,

	// I64Eqz,
	// I64Eq,
	// I64Ne,
	// I64LtS,
	// I64LtU,
	// I64GtS,
	// I64GtU,
	// I64LeS,
	// I64LeU,
	// I64GeS,
	// I64GeU,

	// F32Eq,
	// F32Ne,
	// F32Lt,
	// F32Gt,
	// F32Le,
	// F32Ge,

	// F64Eq,
	// F64Ne,
	// F64Lt,
	// F64Gt,
	// F64Le,
	// F64Ge,

	// I32Clz,
	// I32Ctz,
	// I32Popcnt,
	I32Add,
	I32Sub,
	I32Mul,
	I32DivS,
	I32DivU,
	I32RemS,
	I32RemU,
	I32And,
	I32Or,
	I32Xor,
	I32Shl,
	I32ShrS,
	I32ShrU,
	I32Rotl,
	I32Rotr,
	// I64Clz,
	// I64Ctz,
	// I64Popcnt,
	// I64Add,
	// I64Sub,
	// I64Mul,
	// I64DivS,
	// I64DivU,
	// I64RemS,
	// I64RemU,
	// I64And,
	// I64Or,
	// I64Xor,
	// I64Shl,
	// I64ShrS,
	// I64ShrU,
	// I64Rotl,
	// I64Rotr,
	// F32Abs,
	// F32Neg,
	// F32Ceil,
	// F32Floor,
	// F32Trunc,
	// F32Nearest,
	// F32Sqrt,
	// F32Add,
	// F32Sub,
	// F32Mul,
	// F32Div,
	// F32Min,
	// F32Max,
	// F32Copysign,
	// F64Abs,
	// F64Neg,
	// F64Ceil,
	// F64Floor,
	// F64Trunc,
	// F64Nearest,
	// F64Sqrt,
	// F64Add,
	// F64Sub,
	// F64Mul,
	// F64Div,
	// F64Min,
	// F64Max,
	// F64Copysign,

	// I32WrapI64,
	// I32TruncSF32,
	// I32TruncUF32,
	// I32TruncSF64,
	// I32TruncUF64,
	// I64ExtendSI32,
	// I64ExtendUI32,
	// I64TruncSF32,
	// I64TruncUF32,
	// I64TruncSF64,
	// I64TruncUF64,
	// F32ConvertSI32,
	// F32ConvertUI32,
	// F32ConvertSI64,
	// F32ConvertUI64,
	// F32DemoteF64,
	// F64ConvertSI32,
	// F64ConvertUI32,
	// F64ConvertSI64,
	// F64ConvertUI64,
	// F64PromoteF32,

	// I32ReinterpretF32,
	// I64ReinterpretF64,
	// F32ReinterpretI32,
	// F64ReinterpretI64,

	// I32Extend8S,
	// I32Extend16S,
	// I64Extend8S,
	// I64Extend16S,
	// I64Extend32S,
}

impl Instruction {
	pub fn stack_pops_pushs(&self) -> (&'static [ValueType], &'static [ValueType]) {
		use Instruction::*;

		match self {
			Nop => (&[], &[]),

			I32Const => (&[], &[I32]),

			// itestop
			I32Eqz => (&[I32], &[I32]),
			// irelop
			I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS
			| I32GeU => (&[I32, I32], &[I32]),
			// ibinop
			I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or
			| I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr => (&[I32, I32], &[I32]),

			// parametric
			I32Drop => (&[I32], &[]),
			I32Select => (&[I32, I32, I32], &[I32]),

			// locals
			I32GetLocal => (&[], &[I32]),
			I32SetLocal => (&[I32], &[]),
			I32TeeLocal => (&[], &[]),
		}
	}

	pub fn stack_pop_push_count(&self) -> (usize, usize) {
		let (pops, pushs) = self.stack_pops_pushs();
		(pops.len(), pushs.len())
	}
}

impl From<PInstruction> for Instruction {
	// only to be called with non-parametric instructions
	fn from(pi: PInstruction) -> Self {
		use PInstruction::*;

		match pi {
			Nop => Instruction::Nop,

			Drop | Select | GetLocal(_) | SetLocal(_) | TeeLocal(_) => {
				panic!("Called with parametric instruction")
			}

			I32Const(_) => Instruction::I32Const,

			I32Eqz => Instruction::I32Eqz,
			I32Eq => Instruction::I32Eq,
			I32Ne => Instruction::I32Ne,
			I32LtS => Instruction::I32LtS,
			I32LtU => Instruction::I32LtU,
			I32GtS => Instruction::I32GtS,
			I32GtU => Instruction::I32GtU,
			I32LeS => Instruction::I32LeS,
			I32LeU => Instruction::I32LeU,
			I32GeS => Instruction::I32GeS,
			I32GeU => Instruction::I32GeU,

			I32Add => Instruction::I32Add,
			I32Sub => Instruction::I32Sub,
			I32Mul => Instruction::I32Mul,
			I32DivS => Instruction::I32DivS,
			I32DivU => Instruction::I32DivU,
			I32RemS => Instruction::I32RemS,
			I32RemU => Instruction::I32RemU,
			I32And => Instruction::I32And,
			I32Or => Instruction::I32Or,
			I32Xor => Instruction::I32Xor,
			I32Shl => Instruction::I32Shl,
			I32ShrS => Instruction::I32ShrS,
			I32ShrU => Instruction::I32ShrU,
			I32Rotl => Instruction::I32Rotl,
			I32Rotr => Instruction::I32Rotr,

			_ => unimplemented!(),
		}
	}
}

impl From<Instruction> for PInstruction {
	// only to be called with non-parametric instructions
	fn from(i: Instruction) -> Self {
		use Instruction::*;

		match i {
			Nop => PInstruction::Nop,

			I32Drop => PInstruction::Drop,
			I32Select => PInstruction::Select,
			I32Const | I32GetLocal | I32SetLocal | I32TeeLocal => {
				panic!("Called with parametric instruction")
			}

			I32Eqz => PInstruction::I32Eqz,
			I32Eq => PInstruction::I32Eq,
			I32Ne => PInstruction::I32Ne,
			I32LtS => PInstruction::I32LtS,
			I32LtU => PInstruction::I32LtU,
			I32GtS => PInstruction::I32GtS,
			I32GtU => PInstruction::I32GtU,
			I32LeS => PInstruction::I32LeS,
			I32LeU => PInstruction::I32LeU,
			I32GeS => PInstruction::I32GeS,
			I32GeU => PInstruction::I32GeU,

			I32Add => PInstruction::I32Add,
			I32Sub => PInstruction::I32Sub,
			I32Mul => PInstruction::I32Mul,
			I32DivS => PInstruction::I32DivS,
			I32DivU => PInstruction::I32DivU,
			I32RemS => PInstruction::I32RemS,
			I32RemU => PInstruction::I32RemU,
			I32And => PInstruction::I32And,
			I32Or => PInstruction::I32Or,
			I32Xor => PInstruction::I32Xor,
			I32Shl => PInstruction::I32Shl,
			I32ShrS => PInstruction::I32ShrS,
			I32ShrU => PInstruction::I32ShrU,
			I32Rotl => PInstruction::I32Rotl,
			I32Rotr => PInstruction::I32Rotr,
		}
	}
}

// TODO this assumes stack cannot underflow
pub fn from_parity_wasm_instructions(
	source: &[PInstruction],
	local_types: &[ValueType],
) -> Vec<Instruction> {
	let mut stack_types = Vec::new();
	let mut target = Vec::new();

	for i in source {
		let ins = match i {
			PInstruction::Drop => match stack_types.pop().unwrap() {
				I32 => Instruction::I32Drop,
				_ => unimplemented!(),
			},
			PInstruction::Select => {
				assert_eq!(stack_types.pop().unwrap(), I32);
				let (op1, op2) = (stack_types.pop().unwrap(), stack_types.pop().unwrap());
				assert_eq!(op1, op2);

				stack_types.push(op1);

				match op1 {
					I32 => Instruction::I32Select,
					_ => unimplemented!(),
				}
			}
			PInstruction::GetLocal(index) => {
				let ty = local_types[*index as usize];
				stack_types.push(ty);
				match ty {
					I32 => Instruction::I32GetLocal,
					_ => unimplemented!(),
				}
			}
			PInstruction::SetLocal(index) => {
				let ty = local_types[*index as usize];
				assert_eq!(ty, stack_types.pop().unwrap());
				match ty {
					I32 => Instruction::I32SetLocal,
					_ => unimplemented!(),
				}
			}
			PInstruction::TeeLocal(index) => match local_types[*index as usize] {
				I32 => Instruction::I32TeeLocal,
				_ => unimplemented!(),
			},
			x => {
				let ins = Instruction::from(x.clone());
				let (pops, pushs) = ins.stack_pops_pushs();
				for pop in pops {
					assert_eq!(*pop, stack_types.pop().unwrap());
				}
				stack_types.extend(pushs);
				ins
			}
		};
		target.push(ins);
	}

	target
}

pub fn stack_pop_push_count(i: &PInstruction) -> (usize, usize) {
	use PInstruction::*;

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

pub fn stack_depth(program: &[PInstruction]) -> usize {
	let mut stack_pointer: isize = 0;
	let mut lowest: isize = 0;
	for i in program {
		let (pops, pushs) = stack_pop_push_count(i);
		let (pops, pushs) = (pops as isize, pushs as isize);
		lowest = std::cmp::min(lowest, stack_pointer - pops);
		stack_pointer = stack_pointer - pops + pushs;
	}
	lowest.abs() as usize
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

		for i in Instruction::into_enum_iter() {
			let (pops, pushs) = i.stack_pop_push_count();
			assert_eq!(
				eval(&constants.stack_pop_count(&constants.instruction(&i))),
				pops
			);
			assert_eq!(
				eval(&constants.stack_push_count(&constants.instruction(&i))),
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
			eval(&constants.stack_pop_count(&constants.instruction(&Instruction::I32Const))),
			0
		);
		assert_eq!(
			eval(&constants.stack_push_count(&constants.instruction(&Instruction::I32Const))),
			1
		);
	}

	#[test]
	fn convert_test() {
		let source = {
			use PInstruction::*;
			&[
				I32Const(1),
				I32Const(2),
				I32Const(3),
				Select,
				Drop,
				GetLocal(0),
				TeeLocal(1),
				SetLocal(0),
			]
		};
		let local_types = &[I32, I32];

		let expected = {
			use Instruction::*;
			vec![
				I32Const,
				I32Const,
				I32Const,
				I32Select,
				I32Drop,
				I32GetLocal,
				I32TeeLocal,
				I32SetLocal,
			]
		};
		assert_eq!(expected, from_parity_wasm_instructions(source, local_types));
	}
}
