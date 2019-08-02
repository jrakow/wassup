use crate::{Value, ValueTypeConfig};
use parity_wasm::elements::{Instruction as PInstruction, ValueType};
use z3::*;

#[derive(Clone, Copy, Debug, PartialEq)]
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
	Drop,
	Select,

	GetLocal(u32),
	SetLocal(u32),
	TeeLocal(u32),
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
	Const(Value),

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

	I64Eqz,
	I64Eq,
	I64Ne,
	I64LtS,
	I64LtU,
	I64GtS,
	I64GtU,
	I64LeS,
	I64LeU,
	I64GeS,
	I64GeU,

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
	I64Add,
	I64Sub,
	I64Mul,
	I64DivS,
	I64DivU,
	I64RemS,
	I64RemU,
	I64And,
	I64Or,
	I64Xor,
	I64Shl,
	I64ShrS,
	I64ShrU,
	I64Rotl,
	I64Rotr,
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
	I32WrapI64,
	// I32TruncSF32,
	// I32TruncUF32,
	// I32TruncSF64,
	// I32TruncUF64,
	I64ExtendSI32,
	I64ExtendUI32,
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
	pub fn stack_pop_push_count(&self) -> (usize, usize) {
		use Instruction::*;

		match self {
			Unreachable | Nop=> (0, 0),
			Drop =>  (1, 0),
			Select => (3, 1),

			GetLocal(_) => (0, 1),
			SetLocal(_) => (1, 0),
			TeeLocal(_) => (1, 1),

			Const(_) => (0, 1),

			I32Eqz | I64Eqz => (1, 1),
			// i32 relop
			I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS |I32GeU |
			// i64 relop
			I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS |I64GeU |
			// i64 binop
			I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |
			// i64 binop
			I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr
			=> (2, 1)                                 ,

			// conversions
			I32WrapI64 | I64ExtendSI32 | I64ExtendUI32 => (1, 1),
		}
	}

	pub fn try_convert(pi: &PInstruction) -> Option<Self> {
		use PInstruction::*;

		match pi {
			Unreachable => Some(Instruction::Unreachable),
			Nop => Some(Instruction::Nop),
			Drop => Some(Instruction::Drop),
			Select => Some(Instruction::Select),

			GetLocal(i) => Some(Instruction::GetLocal(*i)),
			SetLocal(i) => Some(Instruction::SetLocal(*i)),
			TeeLocal(i) => Some(Instruction::TeeLocal(*i)),

			I32Const(i) => Some(Instruction::Const(Value::I32(*i))),
			I64Const(i) => Some(Instruction::Const(Value::I64(*i))),

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

			I64Eqz => Some(Instruction::I64Eqz),
			I64Eq => Some(Instruction::I64Eq),
			I64Ne => Some(Instruction::I64Ne),
			I64LtS => Some(Instruction::I64LtS),
			I64LtU => Some(Instruction::I64LtU),
			I64GtS => Some(Instruction::I64GtS),
			I64GtU => Some(Instruction::I64GtU),
			I64LeS => Some(Instruction::I64LeS),
			I64LeU => Some(Instruction::I64LeU),
			I64GeS => Some(Instruction::I64GeS),
			I64GeU => Some(Instruction::I64GeU),

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

			I64Add => Some(Instruction::I64Add),
			I64Sub => Some(Instruction::I64Sub),
			I64Mul => Some(Instruction::I64Mul),
			I64DivS => Some(Instruction::I64DivS),
			I64DivU => Some(Instruction::I64DivU),
			I64RemS => Some(Instruction::I64RemS),
			I64RemU => Some(Instruction::I64RemU),
			I64And => Some(Instruction::I64And),
			I64Or => Some(Instruction::I64Or),
			I64Xor => Some(Instruction::I64Xor),
			I64Shl => Some(Instruction::I64Shl),
			I64ShrS => Some(Instruction::I64ShrS),
			I64ShrU => Some(Instruction::I64ShrU),
			I64Rotl => Some(Instruction::I64Rotl),
			I64Rotr => Some(Instruction::I64Rotr),

			I32WrapI64 => Some(Instruction::I32WrapI64),
			I64ExtendSI32 => Some(Instruction::I64ExtendSI32),
			I64ExtendUI32 => Some(Instruction::I64ExtendUI32),

			_ => None,
		}
	}

	pub fn iter_templates(value_type_config: ValueTypeConfig) -> impl Iterator<Item = Instruction> {
		use Instruction::*;

		let either = &[
			Unreachable,
			Nop,
			Drop,
			Select,
			GetLocal(0),
			SetLocal(0),
			TeeLocal(0),
			Const(Value::I32(0)),
		];
		let i_32 = &[
			I32Eqz, I32Eq, I32Ne, I32LtS, I32LtU, I32GtS, I32GtU, I32LeS, I32LeU, I32GeS, I32GeU,
			I32Add, I32Sub, I32Mul, I32DivS, I32DivU, I32RemS, I32RemU, I32And, I32Or, I32Xor,
			I32Shl, I32ShrS, I32ShrU, I32Rotl, I32Rotr,
		][..];

		let i_64 = if value_type_config.i64_enabled() {
			&[
				I64Eqz,
				I64Eq,
				I64Ne,
				I64LtS,
				I64LtU,
				I64GtS,
				I64GtU,
				I64LeS,
				I64LeU,
				I64GeS,
				I64GeU,
				I64Add,
				I64Sub,
				I64Mul,
				I64DivS,
				I64DivU,
				I64RemS,
				I64RemU,
				I64And,
				I64Or,
				I64Xor,
				I64Shl,
				I64ShrS,
				I64ShrU,
				I64Rotl,
				I64Rotr,
				I32WrapI64,
				I64ExtendSI32,
				I64ExtendUI32,
			][..]
		} else {
			&[]
		};

		either.iter().chain(i_32.iter()).chain(i_64.iter()).cloned()
	}

	pub fn template_eq(&self, other: &Self) -> bool {
		use std::mem::discriminant;

		discriminant(self) == discriminant(other)
	}

	pub fn as_usize(&self, value_type_config: ValueTypeConfig) -> usize {
		Self::iter_templates(value_type_config)
			.position(|i| i.template_eq(self))
			.unwrap()
	}

	pub fn encode<'ctx>(
		&self,
		ctx: &'ctx Context,
		value_type_config: ValueTypeConfig,
	) -> Ast<'ctx> {
		use Instruction::*;

		let constructor = &instruction_datatype(ctx, value_type_config).variants
			[self.as_usize(value_type_config)]
		.constructor;
		match self {
			Const(i) => {
				let ty = i.value_type();
				constructor.apply(&[
					&i.encode(ctx, value_type_config),
					&value_type_config.encode_value_type(ctx, ty),
				])
			}
			GetLocal(i) | SetLocal(i) | TeeLocal(i) => constructor.apply(&[&ctx.from_u32(*i)]),
			_ => constructor.apply(&[]),
		}
	}

	pub fn decode(
		encoded_instr: &Ast,
		ctx: &Context,
		model: &Model,
		value_type_config: ValueTypeConfig,
	) -> Self {
		use Instruction::*;

		Instruction::iter_templates(value_type_config)
			.find_map(|template| {
				let variant = &instruction_datatype(ctx, value_type_config).variants
					[template.as_usize(value_type_config)];

				let active = variant.tester.apply(&[&encoded_instr]);
				if model.eval(&active).unwrap().as_bool().unwrap() {
					Some(match template {
						Const(_) => {
							let ty = variant.accessors[1].apply(&[&encoded_instr]);
							let ty = value_type_config.decode_value_type(ctx, model, &ty);
							let encoded_value = variant.accessors[0].apply(&[&encoded_instr]);
							let value = Value::decode(&encoded_value, model, ty, value_type_config);

							Const(value)
						}
						GetLocal(_) => {
							let ast = variant.accessors[0].apply(&[&encoded_instr]);
							let v = model.eval(&ast).unwrap().as_u32().unwrap();
							GetLocal(v)
						}
						SetLocal(_) => {
							let ast = variant.accessors[0].apply(&[&encoded_instr]);
							let v = model.eval(&ast).unwrap().as_u32().unwrap();
							SetLocal(v)
						}
						TeeLocal(_) => {
							let ast = variant.accessors[0].apply(&[&encoded_instr]);
							let v = model.eval(&ast).unwrap().as_u32().unwrap();
							TeeLocal(v)
						}
						x => x,
					})
				} else {
					None
				}
			})
			.unwrap()
	}
}

impl From<Instruction> for PInstruction {
	/// Convert to parity-wasm's instruction
	fn from(i: Instruction) -> Self {
		use Instruction::*;

		match i {
			Unreachable => PInstruction::Unreachable,
			Nop => PInstruction::Nop,

			Drop => PInstruction::Drop,
			Select => PInstruction::Select,

			Const(Value::I32(i)) => PInstruction::I32Const(i),
			Const(Value::I64(i)) => PInstruction::I64Const(i),
			Const(_) => unimplemented!(),
			GetLocal(i) => PInstruction::GetLocal(i),
			SetLocal(i) => PInstruction::SetLocal(i),
			TeeLocal(i) => PInstruction::TeeLocal(i),

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

			I64Eqz => PInstruction::I64Eqz,
			I64Eq => PInstruction::I64Eq,
			I64Ne => PInstruction::I64Ne,
			I64LtS => PInstruction::I64LtS,
			I64LtU => PInstruction::I64LtU,
			I64GtS => PInstruction::I64GtS,
			I64GtU => PInstruction::I64GtU,
			I64LeS => PInstruction::I64LeS,
			I64LeU => PInstruction::I64LeU,
			I64GeS => PInstruction::I64GeS,
			I64GeU => PInstruction::I64GeU,

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

			I64Add => PInstruction::I64Add,
			I64Sub => PInstruction::I64Sub,
			I64Mul => PInstruction::I64Mul,
			I64DivS => PInstruction::I64DivS,
			I64DivU => PInstruction::I64DivU,
			I64RemS => PInstruction::I64RemS,
			I64RemU => PInstruction::I64RemU,
			I64And => PInstruction::I64And,
			I64Or => PInstruction::I64Or,
			I64Xor => PInstruction::I64Xor,
			I64Shl => PInstruction::I64Shl,
			I64ShrS => PInstruction::I64ShrS,
			I64ShrU => PInstruction::I64ShrU,
			I64Rotl => PInstruction::I64Rotl,
			I64Rotr => PInstruction::I64Rotr,

			I32WrapI64 => PInstruction::I32WrapI64,
			I64ExtendSI32 => PInstruction::I64ExtendSI32,
			I64ExtendUI32 => PInstruction::I64ExtendUI32,
		}
	}
}

/// Datatype for instructions in Z3
///
/// Instructions are indexed according to their enum discriminant.
pub fn instruction_datatype(ctx: &Context, value_type_config: ValueTypeConfig) -> Datatype {
	let mut datatype = DatatypeBuilder::new(ctx);
	let value_sort = value_type_config.value_sort(ctx);
	let value_type_datatype = value_type_config.value_type_datatype(ctx);

	for i in Instruction::iter_templates(value_type_config) {
		datatype = match i {
			Instruction::Const(_) => datatype.variant(
				"Const",
				&[
					("value", &value_sort),
					("value_type", &value_type_datatype.sort),
				],
			),
			Instruction::GetLocal(_) => {
				datatype.variant("GetLocal", &[("get_local_index", &ctx.int_sort())])
			}
			Instruction::SetLocal(_) => {
				datatype.variant("SetLocal", &[("set_local_index", &ctx.int_sort())])
			}
			Instruction::TeeLocal(_) => {
				datatype.variant("TeeLocal", &[("tee_local_index", &ctx.int_sort())])
			}
			x => {
				let name = format!("{:?}", x);
				datatype.variant(&name, &[])
			}
		}
	}

	datatype.finish("Instruction")
}

// What types the program assumes to have on the stack
//
// This cannot be computed from just the snippet because of paramtric instructions.
// This will panic on a select where the operand types is unknown.
pub fn initial_stack_types(program: &[Instruction], local_types: &[ValueType]) -> Vec<ValueType> {
	use Instruction::*;
	use ValueType::*;

	let mut initial_stack: Vec<ValueType> = Default::default();
	let mut stack: Vec<ValueType> = Default::default();

	for ins in program {
		let (pops, pushs) = match ins {
			// special case for parametric instructions
			Drop => {
				if stack.is_empty() {
					// assume I32
					initial_stack.push(I32);
				} else {
					stack.pop().unwrap();
				}

				continue;
			}
			Select => {
				if stack.is_empty() {
					initial_stack.push(I32);
				} else {
					stack.pop().unwrap();
				}

				if stack.is_empty() {
					panic!(
						"Insufficient information to compute stack types for snippet {:?}",
						&program
					);
				} else if stack.len() == 1 {
					// type of initial stack same as type of operand 1
					initial_stack.push(stack.pop().unwrap())
				} else {
					stack.pop().unwrap();
					stack.pop().unwrap();
				}

				continue;
			}

			// normal cases
			Unreachable => (vec![], vec![]),
			Nop => (vec![], vec![]),

			Const(Value::I32(_)) => (vec![], vec![I32]),
			Const(Value::I64(_)) => (vec![], vec![I64]),
			Const(_) => unimplemented!(),

			// itestop
			I32Eqz => (vec![I32], vec![I32]),
			I64Eqz => (vec![I64], vec![I64]),
			// irelop
			I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS
			| I32GeU => (vec![I32, I32], vec![I32]),
			I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS
			| I64GeU => (vec![I64, I64], vec![I32]),
			// ibinop
			I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or
			| I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr => (vec![I32, I32], vec![I32]),
			I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or
			| I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr => (vec![I64, I64], vec![I64]),

			// conversions
			I32WrapI64 => (vec![I64], vec![I32]),
			I64ExtendSI32 | I64ExtendUI32 => (vec![I32], vec![I64]),

			// locals
			GetLocal(i) => (vec![], vec![local_types[*i as usize]]),
			SetLocal(i) => (vec![local_types[*i as usize]], vec![]),
			TeeLocal(i) => (
				vec![local_types[*i as usize]],
				vec![local_types[*i as usize]],
			),
		};

		for ty in pops {
			if stack.is_empty() {
				initial_stack.push(ty);
			} else {
				stack.pop().unwrap();
			}
		}
		stack.extend(pushs);
	}

	initial_stack
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::Constants;
	use Value::*;

	#[test]
	fn test_stack_pop_push_count() {
		let ctx = {
			let cfg = Config::default();
			Context::new(&cfg)
		};
		let solver = Solver::new(&ctx);
		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};
		let constants = Constants::new(
			&ctx,
			&solver,
			vec![],
			vec![],
			vec![],
			&[],
			value_type_config,
		);

		assert!(solver.check());
		let model = solver.get_model();

		let eval = |ast: &Ast| -> usize {
			let ast = model.eval(ast).unwrap();
			ast.as_usize().unwrap()
		};

		for i in Instruction::iter_templates(value_type_config) {
			let (pops, pushs) = i.stack_pop_push_count();
			assert_eq!(
				eval(&constants.stack_pop_count(&i.encode(&ctx, value_type_config))),
				pops
			);
			assert_eq!(
				eval(&constants.stack_push_count(&i.encode(&ctx, value_type_config))),
				pushs
			);
		}

		assert_eq!(
			eval(&constants.stack_pop_count(&Instruction::I32Add.encode(&ctx, value_type_config))),
			2
		);
		assert_eq!(
			eval(&constants.stack_push_count(&Instruction::I32Add.encode(&ctx, value_type_config))),
			1
		);
		assert_eq!(
			eval(
				&constants
					.stack_pop_count(&Instruction::Const(I32(0)).encode(&ctx, value_type_config))
			),
			0
		);
		assert_eq!(
			eval(
				&constants
					.stack_push_count(&Instruction::Const(I32(0)).encode(&ctx, value_type_config))
			),
			1
		);
	}

	#[test]
	fn test_initial_stack_types() {
		use Instruction::*;
		use ValueType::*;

		assert_eq!(&initial_stack_types(&[], &[]), &[],);

		assert_eq!(
			&initial_stack_types(&[Const(Value::I32(0)), Const(Value::I32(1)), I32Add], &[]),
			&[],
		);

		assert_eq!(
			&initial_stack_types(&[Const(Value::I32(0)), I32Add], &[]),
			&[I32],
		);

		assert_eq!(
			&initial_stack_types(&[Const(Value::I64(0)), Const(Value::I32(0)), Select], &[]),
			&[I64],
		);

		assert_eq!(
			&initial_stack_types(&[Const(Value::I32(0)), I64ExtendUI32, I64Add], &[]),
			&[I64],
		);

		assert_eq!(
			&initial_stack_types(&[Const(Value::I64(0)), I32WrapI64, I32Add], &[]),
			&[I32],
		);

		assert_eq!(
			&initial_stack_types(&[SetLocal(0), TeeLocal(1)], &[I32, I64]),
			&[I32, I64],
		);
	}
}
