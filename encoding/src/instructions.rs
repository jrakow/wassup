use parity_wasm::elements::{
	Instruction as PInstruction,
	ValueType::{self, *},
};
use z3::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Directly encodable Wasm instruction.
///
/// There a few noteworthy differences to parity-wasm's instructions:
/// - `I32Drop`/`I32Select` are typed, not parametric like `Drop`
/// - Not all instructions are implemented
pub enum Instruction {
	Unreachable,
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

	I32GetLocal(u32),
	I32SetLocal(u32),
	I32TeeLocal(u32),
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
	I32Const(i32),
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
}

impl Instruction {
	/// Parameters and returns of the instruction
	pub fn stack_pops_pushs(&self) -> (&'static [ValueType], &'static [ValueType]) {
		use Instruction::*;

		match self {
			Unreachable => (&[], &[]),
			Nop => (&[], &[]),

			I32Const(_) => (&[], &[I32]),

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
			I32GetLocal(_) => (&[], &[I32]),
			I32SetLocal(_) => (&[I32], &[]),
			I32TeeLocal(_) => (&[I32], &[I32]),
		}
	}

	pub fn stack_pop_push_count(&self) -> (usize, usize) {
		let (pops, pushs) = self.stack_pops_pushs();
		(pops.len(), pushs.len())
	}

	pub fn try_convert(pi: &PInstruction) -> Option<Self> {
		use PInstruction::*;

		match pi {
			Unreachable => Some(Instruction::Unreachable),
			Nop => Some(Instruction::Nop),

			I32Const(i) => Some(Instruction::I32Const(*i)),

			I32Eqz => Some(Instruction::I32Eqz),
			I32Eq => Some(Instruction::I32Eq),
			I32Ne => Some(Instruction::I32Ne),
			I32LtS => Some(Instruction::I32LtS),
			I32LtU => Some(Instruction::I32LtU),
			I32GtS => Some(Instruction::I32GtS),
			I32GtU => Some(Instruction::I32GtU),
			I32LeS => Some(Instruction::I32LeS),
			I32LeU => Some(Instruction::I32LeU),
			I32GeS => Some(Instruction::I32GeS),
			I32GeU => Some(Instruction::I32GeU),

			I32Add => Some(Instruction::I32Add),
			I32Sub => Some(Instruction::I32Sub),
			I32Mul => Some(Instruction::I32Mul),
			I32DivS => Some(Instruction::I32DivS),
			I32DivU => Some(Instruction::I32DivU),
			I32RemS => Some(Instruction::I32RemS),
			I32RemU => Some(Instruction::I32RemU),
			I32And => Some(Instruction::I32And),
			I32Or => Some(Instruction::I32Or),
			I32Xor => Some(Instruction::I32Xor),
			I32Shl => Some(Instruction::I32Shl),
			I32ShrS => Some(Instruction::I32ShrS),
			I32ShrU => Some(Instruction::I32ShrU),
			I32Rotl => Some(Instruction::I32Rotl),
			I32Rotr => Some(Instruction::I32Rotr),

			_ => None,
		}
	}

	pub fn iter_templates() -> impl Iterator<Item = Instruction> {
		use Instruction::*;
		[
			Unreachable,
			Nop,
			I32Drop,
			I32Select,
			I32GetLocal(0),
			I32SetLocal(0),
			I32TeeLocal(0),
			I32Const(0),
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
		]
		.iter()
		.cloned()
	}

	pub fn template_eq(&self, other: &Self) -> bool {
		use std::mem::discriminant;

		discriminant(self) == discriminant(other)
	}

	pub fn as_usize(&self) -> usize {
		Self::iter_templates()
			.position(|i| i.template_eq(self))
			.unwrap()
	}

	pub fn encode<'ctx>(&self, ctx: &'ctx Context) -> Ast<'ctx> {
		use Instruction::*;

		let constructor = &instruction_datatype(ctx).variants[self.as_usize()].constructor;
		match self {
			I32Const(i) => constructor.apply(&[&ctx.from_i32(*i).int2bv(32)]),
			I32GetLocal(i) | I32SetLocal(i) | I32TeeLocal(i) => {
				constructor.apply(&[&ctx.from_u32(*i)])
			}
			_ => constructor.apply(&[]),
		}
	}
}

impl From<Instruction> for PInstruction {
	/// Convert to parity-wasm's instruction
	fn from(i: Instruction) -> Self {
		use Instruction::*;

		match i {
			Unreachable => PInstruction::Unreachable,
			Nop => PInstruction::Nop,

			I32Drop => PInstruction::Drop,
			I32Select => PInstruction::Select,

			I32Const(i) => PInstruction::I32Const(i),
			I32GetLocal(i) => PInstruction::GetLocal(i),
			I32SetLocal(i) => PInstruction::SetLocal(i),
			I32TeeLocal(i) => PInstruction::TeeLocal(i),

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

/// Datatype for instructions in Z3
///
/// Instructions are indexed according to their enum discriminant.
pub fn instruction_datatype(ctx: &Context) -> Datatype {
	let mut datatype = DatatypeBuilder::new(ctx);

	let bv_32 = ctx.bitvector_sort(32);
	for i in Instruction::iter_templates() {
		datatype = match i {
			Instruction::I32Const(_) => datatype.variant("I32Const", &[("value", &bv_32)]),
			Instruction::I32GetLocal(_) => {
				datatype.variant("I32GetLocal", &[("get_local_index", &ctx.int_sort())])
			}
			Instruction::I32SetLocal(_) => {
				datatype.variant("I32SetLocal", &[("set_local_index", &ctx.int_sort())])
			}
			Instruction::I32TeeLocal(_) => {
				datatype.variant("I32TeeLocal", &[("tee_local_index", &ctx.int_sort())])
			}
			x => {
				let name = format!("{:?}", x);
				datatype.variant(&name, &[])
			}
		}
	}

	datatype.finish("Instruction")
}

/// How many words this instructions sequence assumes to be on the stack, when it starts executing.
pub fn stack_depth(program: &[Instruction]) -> usize {
	let mut stack_pointer: isize = 0;
	let mut lowest: isize = 0;
	for i in program {
		let (pops, pushs) = i.stack_pop_push_count();
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

	#[test]
	fn test_stack_pop_push_count() {
		let ctx = {
			let cfg = Config::default();
			Context::new(&cfg)
		};
		let solver = Solver::new(&ctx);
		let constants = Constants::new(&ctx, vec![], &[]);

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: &Ast| -> usize {
			let ast = model.eval(ast).unwrap();
			ast.as_usize().unwrap()
		};

		for i in Instruction::iter_templates() {
			let (pops, pushs) = i.stack_pop_push_count();
			assert_eq!(eval(&constants.stack_pop_count(&i.encode(&ctx))), pops);
			assert_eq!(eval(&constants.stack_push_count(&i.encode(&ctx))), pushs);
		}

		assert_eq!(
			eval(&constants.stack_pop_count(&Instruction::I32Add.encode(&ctx))),
			2
		);
		assert_eq!(
			eval(&constants.stack_push_count(&Instruction::I32Add.encode(&ctx))),
			1
		);
		assert_eq!(
			eval(&constants.stack_pop_count(&Instruction::I32Const(0).encode(&ctx))),
			0
		);
		assert_eq!(
			eval(&constants.stack_push_count(&Instruction::I32Const(0).encode(&ctx))),
			1
		);
	}
}
