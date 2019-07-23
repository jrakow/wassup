use crate::{instructions::Instruction, Value};
use either::Either;
use parity_wasm::elements::{
	FuncBody, Instruction as PInstruction, Instructions, Local, ValueType,
};

#[derive(Clone, Debug, PartialEq)]
/// Equivalent of a Wasm function definition
pub struct Function {
	/// Types of the parameters and locals
	/// The first n_params are parameters, the rest locals
	pub local_types: Vec<ValueType>,
	/// Number of parameters
	///
	/// Always <= local_types.len()
	pub n_params: usize,
	/// The instructions of the function.
	///
	/// Split into snippets of encodable and unencodable instructions
	pub instructions: Vec<Either<Vec<Instruction>, PInstruction>>,
}

impl Function {
	/// Create a function
	pub fn from_wasm_func_body_params(body: &FuncBody, params: &[ValueType]) -> Self {
		use Instruction::*;
		use ValueType::*;

		let local_types = {
			let mut v = params.to_vec();
			for l in body.locals() {
				for _ in 0..l.count() {
					v.push(l.value_type());
				}
			}
			v
		};
		let wasm_instructions = body.code().elements();

		// stack type in each execution step
		// start with empty stack
		let mut stack_type: Vec<ValueType> = Vec::new();

		let mut instructions = Vec::new();
		for wasm_instruction in wasm_instructions.iter() {
			let either = if let Some(ins) = Instruction::try_convert(wasm_instruction) {
				let (pops, pushs): (Vec<ValueType>, Vec<ValueType>) = match ins {
					Unreachable => (vec![], vec![]),
					Nop => (vec![], vec![]),

					Const(Value::I32(_)) => (vec![], vec![I32]),
					Const(Value::I64(_)) => (vec![], vec![I64]),
					Const(_) => unimplemented!(),

					// itestop
					I32Eqz => (vec![I32], vec![I32]),
					I64Eqz => (vec![I64], vec![I64]),
					// irelop
					I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU
					| I32GeS | I32GeU => (vec![I32, I32], vec![I32]),
					I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU
					| I64GeS | I64GeU => (vec![I64, I64], vec![I32]),
					// ibinop
					I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And
					| I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr => {
						(vec![I32, I32], vec![I32])
					}
					I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And
					| I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr => {
						(vec![I64, I64], vec![I64])
					}

					// conversions
					I32WrapI64 => (vec![I64], vec![I32]),
					I64ExtendSI32 | I64ExtendUI32 => (vec![I32], vec![I64]),

					// parametric
					Drop => (vec![stack_type[stack_type.len() - 1]], vec![]),
					Select => (
						vec![
							stack_type[stack_type.len() - 2],
							stack_type[stack_type.len() - 3],
							I32,
						],
						vec![stack_type[stack_type.len() - 2]],
					),

					// locals
					GetLocal(i) => (vec![], vec![local_types[i as usize]]),
					SetLocal(i) => (vec![local_types[i as usize]], vec![]),
					TeeLocal(i) => (vec![local_types[i as usize]], vec![local_types[i as usize]]),
				};
				for _ in pops {
					stack_type.pop().unwrap();
				}
				stack_type.extend(pushs);

				Either::Left(ins)
			} else {
				// TODO change stack types
				Either::Right(wasm_instruction.clone())
			};
			instructions.push(either);
		}

		// Gather encodable instructions into vector
		let instructions = gather_encodable_instructions(&instructions);

		Self {
			local_types,
			n_params: params.len(),
			instructions,
		}
	}

	pub fn to_wasm_func_body(&self) -> FuncBody {
		let locals = gather_locals(&self.local_types);

		let mut wasm_instructions = Vec::new();
		for instr in self.instructions.iter() {
			match instr {
				Either::Left(vec) => {
					for i in vec {
						wasm_instructions.push(PInstruction::from(*i))
					}
				}
				Either::Right(i) => wasm_instructions.push(i.clone()),
			};
		}

		FuncBody::new(locals, Instructions::new(wasm_instructions))
	}
}

fn gather_encodable_instructions(
	instructions: &[Either<Instruction, PInstruction>],
) -> Vec<Either<Vec<Instruction>, PInstruction>> {
	let mut acc = Vec::new();

	let mut current_snippet = Vec::new();
	for i in instructions {
		match i {
			// encodable
			Either::Left(i) => current_snippet.push(*i),
			Either::Right(i) => {
				if !current_snippet.is_empty() {
					acc.push(Either::Left(current_snippet));
					current_snippet = Vec::new();
				}
				acc.push(Either::Right(i.clone()));
			}
		}
	}

	if !current_snippet.is_empty() {
		acc.push(Either::Left(current_snippet));
	}

	acc
}

fn gather_locals(local_types: &[ValueType]) -> Vec<Local> {
	if local_types.is_empty() {
		Vec::new()
	} else {
		let mut acc = Vec::new();
		let mut current_ty = local_types[0];
		let mut current_count = 0;

		for ty in local_types {
			if *ty == current_ty {
				current_count += 1;
			} else {
				acc.push(Local::new(current_count, current_ty));
				// reset
				current_ty = *ty;
				current_count = 1;
			}
		}

		if current_count > 0 {
			acc.push(Local::new(current_count, current_ty));
		}

		acc
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use parity_wasm::elements::Instructions;
	use Value::*;

	#[test]
	fn convert_test() {
		let source = vec![
			PInstruction::I32Const(1),
			PInstruction::I32Const(2),
			PInstruction::I32Const(3),
			PInstruction::Select,
			PInstruction::Drop,
			PInstruction::GetLocal(0),
			PInstruction::TeeLocal(1),
			PInstruction::SetLocal(0),
		];
		let locals = vec![Local::new(2, ValueType::I32)];
		let func_body = FuncBody::new(locals, Instructions::new(source));

		let expected = Function {
			local_types: vec![ValueType::I32; 2],
			n_params: 0,
			instructions: vec![Either::Left(vec![
				Instruction::Const(I32(1)),
				Instruction::Const(I32(2)),
				Instruction::Const(I32(3)),
				Instruction::Select,
				Instruction::Drop,
				Instruction::GetLocal(0),
				Instruction::TeeLocal(1),
				Instruction::SetLocal(0),
			])],
		};
		assert_eq!(
			expected,
			Function::from_wasm_func_body_params(&func_body, &[]),
		);
	}

	#[test]
	fn test_gather_locals() {
		use ValueType::*;

		let local_types = &[I32, I32, I64, I32];
		let locals = vec![Local::new(2, I32), Local::new(1, I64), Local::new(1, I32)];
		assert_eq!(locals, gather_locals(local_types));

		let local_types = &[];
		let locals: Vec<Local> = vec![];
		assert_eq!(locals, gather_locals(local_types));
	}
}
