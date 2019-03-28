use parity_wasm::elements::BlockType;
use parity_wasm::elements::Instruction;
use std::mem::replace;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Block {
	Flat(Vec<Instruction>),
	BlockIns {
		ty: BlockType,
		inner: Vec<Block>,
	},
	LoopIns {
		ty: BlockType,
		inner: Vec<Block>,
	},
	IfIns {
		ty: BlockType,
		inner_true: Vec<Block>,
		inner_false: Vec<Block>,
	},
}

fn instruction_starts_block(i: &Instruction) -> bool {
	match i {
		Instruction::Block(_) | Instruction::Loop(_) | Instruction::If(_) => true,
		_ => false,
	}
}

pub fn split_into_blocks(source: &[Instruction]) -> (Vec<Block>, &[Instruction]) {
	let mut source = source;
	let mut blocks = vec![];

	// if the current block is flat, instructions of current block
	let mut ins = vec![];

	while !source.is_empty() {
		let i = &source[0];
		source = &source[1..];
		if !ins.is_empty() && instruction_starts_block(i) {
			blocks.push(Block::Flat(replace(&mut ins, vec![])));
		}

		match i {
			Instruction::Block(ty) => {
				let (inner, rest) = split_into_blocks(source);
				source = rest;
				blocks.push(Block::BlockIns { ty: *ty, inner });
			}
			Instruction::Loop(ty) => {
				let (inner, rest) = split_into_blocks(source);
				source = rest;
				blocks.push(Block::LoopIns { ty: *ty, inner });
			}
			Instruction::If(ty) => {
				let (inner_true, rest) = split_into_blocks(source);
				source = rest;
				let (inner_false, rest) = split_into_blocks(source);
				source = rest;
				blocks.push(Block::IfIns {
					ty: *ty,
					inner_true,
					inner_false,
				});
			}
			Instruction::Else | Instruction::End => {
				break;
			}
			Instruction::Unreachable => {
				ins.push(i.clone());
				break;
			}
			Instruction::Br(_)
			| Instruction::BrIf(_)
			| Instruction::BrTable(_)
			| Instruction::Return
			| Instruction::Call(_)
			| Instruction::CallIndirect(..) => unimplemented!(),
			_ => {
				ins.push(i.clone());
			}
		}
	}

	if !ins.is_empty() {
		blocks.push(Block::Flat(ins));
	}

	(blocks, source)
}

#[cfg(test)]
mod tests {
	use super::*;
	use parity_wasm::elements::BlockType;
	use parity_wasm::elements::Instruction;

	#[test]
	fn nothing_to_split() {
		assert_eq!(
			(
				vec![Block::Flat(vec![
					Instruction::I32Add,
					Instruction::I32Sub,
					Instruction::Drop
				])],
				&[][..]
			),
			split_into_blocks(&[Instruction::I32Add, Instruction::I32Sub, Instruction::Drop]),
		);

		assert_eq!(
			(
				vec![Block::Flat(vec![
					Instruction::I32Add,
					Instruction::Unreachable
				])],
				&[][..]
			),
			split_into_blocks(&[Instruction::I32Add, Instruction::Unreachable]),
		);
	}

	#[test]
	fn block() {
		assert_eq!(
			(
				vec![Block::BlockIns {
					ty: BlockType::NoResult,
					inner: vec![Block::Flat(vec![Instruction::I32Add])],
				}],
				&[][..]
			),
			split_into_blocks(&[
				Instruction::Block(BlockType::NoResult),
				Instruction::I32Add,
				Instruction::End
			]),
		);
		assert_eq!(
			(
				vec![
					Block::Flat(vec![Instruction::I32Add]),
					Block::BlockIns {
						ty: BlockType::NoResult,
						inner: vec![Block::Flat(vec![Instruction::I32Add])],
					},
					Block::Flat(vec![Instruction::I32Sub]),
				],
				&[][..]
			),
			split_into_blocks(&[
				Instruction::I32Add,
				Instruction::Block(BlockType::NoResult),
				Instruction::I32Add,
				Instruction::End,
				Instruction::I32Sub
			]),
		);
	}

	#[test]
	fn if_then_else() {
		assert_eq!(
			(
				vec![Block::IfIns {
					ty: BlockType::NoResult,
					inner_true: vec![Block::Flat(vec![Instruction::I32Sub])],
					inner_false: vec![Block::Flat(vec![Instruction::I32Add])],
				},],
				&[][..]
			),
			split_into_blocks(&[
				Instruction::If(BlockType::NoResult),
				Instruction::I32Sub,
				Instruction::Else,
				Instruction::I32Add,
				Instruction::End
			]),
		);

		assert_eq!(
			(
				vec![
					Block::Flat(vec![Instruction::I32Add]),
					Block::IfIns {
						ty: BlockType::NoResult,
						inner_true: vec![Block::Flat(vec![Instruction::I32Sub])],
						inner_false: vec![Block::Flat(vec![Instruction::I32Add])],
					},
					Block::Flat(vec![Instruction::I32Sub]),
				],
				&[][..]
			),
			split_into_blocks(&[
				Instruction::I32Add,
				Instruction::If(BlockType::NoResult),
				Instruction::I32Sub,
				Instruction::Else,
				Instruction::I32Add,
				Instruction::End,
				Instruction::I32Sub
			]),
		);
	}
}
