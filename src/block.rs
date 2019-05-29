use parity_wasm::elements::BlockType;
use parity_wasm::elements::Instruction;

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
	Unencodable(Instruction),
}

impl Block {
	pub fn serialize(&self, acc: &mut Vec<Instruction>) {
		use crate::block::Block::*;

		match self {
			Flat(v) => acc.extend_from_slice(v),
			BlockIns { ty, inner } => {
				acc.push(Instruction::Block(*ty));
				for i in inner {
					i.serialize(acc);
				}
				acc.push(Instruction::End);
			}
			LoopIns { ty, inner } => {
				acc.push(Instruction::Loop(*ty));
				for i in inner {
					i.serialize(acc);
				}
				acc.push(Instruction::End);
			}
			IfIns {
				ty,
				inner_true,
				inner_false,
			} => {
				acc.push(Instruction::If(*ty));
				for i in inner_true {
					i.serialize(acc);
				}
				acc.push(Instruction::Else);
				for i in inner_false {
					i.serialize(acc);
				}
				acc.push(Instruction::End);
			}
			Unencodable(i) => {
				acc.push(i.clone());
			}
		}
	}
}

pub fn serialize(blocks: &[Block]) -> Vec<Instruction> {
	let mut v = vec![];
	for b in blocks {
		b.serialize(&mut v);
	}
	v.push(Instruction::End);
	v
}

fn instruction_ends_block(i: &Instruction) -> bool {
	match i {
		Instruction::Else | Instruction::End => true,
		_ => false,
	}
}

fn instruction_encodable(i: &Instruction) -> bool {
	match i {
		Instruction::I32Const(_) | Instruction::Nop | Instruction::I32Add => true,
		_ => false,
	}
}

struct ParseState<'i> {
	rest: &'i [Instruction],
}

impl<'i> ParseState<'i> {
	fn new(source: &'i [Instruction]) -> Self {
		Self { rest: source }
	}

	fn peek(&self) -> &'i Instruction {
		&self.rest[0]
	}

	fn advance(&mut self) -> &'i Instruction {
		let (i, rest) = self.rest.split_first().unwrap();
		self.rest = rest;
		i
	}

	// Parses a Block::Flat
	fn parse_flat(&mut self) -> Block {
		let mut acc = vec![];

		while !self.rest.is_empty() {
			match self.peek() {
				i if instruction_encodable(i) => acc.push(self.advance().clone()),
				_ => return Block::Flat(acc),
			}
		}

		unreachable!();
	}

	// Parses a Block
	fn parse_blocks(&mut self) -> Vec<Block> {
		let mut acc = vec![];

		while !self.rest.is_empty() {
			let next_block = match self.peek() {
				i if instruction_ends_block(i) => {
					self.advance();
					return acc;
				}
				Instruction::Block(ty) => {
					self.advance();
					let inner = self.parse_blocks();
					Block::BlockIns { ty: *ty, inner }
				}
				Instruction::Loop(ty) => {
					self.advance();
					let inner = self.parse_blocks();
					Block::LoopIns { ty: *ty, inner }
				}
				Instruction::If(ty) => {
					self.advance();
					let inner_true = self.parse_blocks();
					let inner_false = self.parse_blocks();
					Block::IfIns {
						ty: *ty,
						inner_true,
						inner_false,
					}
				}
				i if instruction_encodable(i) => self.parse_flat(),
				_ => Block::Unencodable(self.advance().clone()),
			};
			acc.push(next_block);
		}
		unreachable!();
	}
}

pub fn parse_blocks(source: &[Instruction]) -> Vec<Block> {
	ParseState::new(source).parse_blocks()
}

fn flat_blocks_mut_impl<'a>(block: &'a mut Block, flat_blocks: &mut Vec<&'a mut Vec<Instruction>>) {
	match block {
		Block::Flat(ins) => {
			flat_blocks.push(ins);
		}
		Block::BlockIns { inner, .. } => {
			for b in inner {
				flat_blocks_mut_impl(b, flat_blocks);
			}
		}
		Block::LoopIns { inner, .. } => {
			for b in inner {
				flat_blocks_mut_impl(b, flat_blocks);
			}
		}
		Block::IfIns {
			inner_true,
			inner_false,
			..
		} => {
			for b in inner_true {
				flat_blocks_mut_impl(b, flat_blocks);
			}
			for b in inner_false {
				flat_blocks_mut_impl(b, flat_blocks);
			}
		}
		Block::Unencodable(_) => {}
	}
}

pub fn flat_blocks_mut(blocks: &mut Vec<Block>) -> Vec<&mut Vec<Instruction>> {
	let mut flat_blocks = vec![];
	for b in blocks {
		flat_blocks_mut_impl(b, &mut flat_blocks);
	}
	flat_blocks
}

#[cfg(test)]
mod tests {
	use super::*;
	use parity_wasm::elements::BlockType;
	use parity_wasm::elements::Instruction;

	#[test]
	fn nothing_to_split() {
		let source = &[
			Instruction::I32Add,
			Instruction::I32Add,
			Instruction::Drop,
			Instruction::End,
		];
		let expected = &[
			Block::Flat(vec![Instruction::I32Add, Instruction::I32Add]),
			Block::Unencodable(Instruction::Drop),
		];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);

		let source = &[
			Instruction::I32Add,
			Instruction::Unreachable,
			Instruction::End,
		];
		let expected = &[
			Block::Flat(vec![Instruction::I32Add]),
			Block::Unencodable(Instruction::Unreachable),
		];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);
	}

	#[test]
	fn empty() {
		let source = &[Instruction::End];
		let expected = &[];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);
	}

	#[test]
	fn end_at_end() {
		let source = &[
			Instruction::I32Add,
			Instruction::Unreachable,
			Instruction::End,
		];
		let expected = &[
			Block::Flat(vec![Instruction::I32Add]),
			Block::Unencodable(Instruction::Unreachable),
		];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);
	}

	#[test]
	fn block() {
		let source = &[
			Instruction::Block(BlockType::NoResult),
			Instruction::I32Add,
			Instruction::End,
			Instruction::End,
		];
		let expected = &[Block::BlockIns {
			ty: BlockType::NoResult,
			inner: vec![Block::Flat(vec![Instruction::I32Add])],
		}];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);

		let source = &[
			Instruction::I32Add,
			Instruction::Block(BlockType::NoResult),
			Instruction::I32Add,
			Instruction::End,
			Instruction::I32Add,
			Instruction::End,
		];
		let expected = &[
			Block::Flat(vec![Instruction::I32Add]),
			Block::BlockIns {
				ty: BlockType::NoResult,
				inner: vec![Block::Flat(vec![Instruction::I32Add])],
			},
			Block::Flat(vec![Instruction::I32Add]),
		];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);
	}

	#[test]
	fn if_then_else() {
		let source = &[
			Instruction::If(BlockType::NoResult),
			Instruction::I32Add,
			Instruction::Else,
			Instruction::I32Add,
			Instruction::End,
			Instruction::End,
		];
		let expected = &[Block::IfIns {
			ty: BlockType::NoResult,
			inner_true: vec![Block::Flat(vec![Instruction::I32Add])],
			inner_false: vec![Block::Flat(vec![Instruction::I32Add])],
		}];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);

		let source = &[
			Instruction::I32Add,
			Instruction::If(BlockType::NoResult),
			Instruction::I32Add,
			Instruction::Else,
			Instruction::I32Add,
			Instruction::End,
			Instruction::I32Add,
			Instruction::End,
		];
		let expected = &[
			Block::Flat(vec![Instruction::I32Add]),
			Block::IfIns {
				ty: BlockType::NoResult,
				inner_true: vec![Block::Flat(vec![Instruction::I32Add])],
				inner_false: vec![Block::Flat(vec![Instruction::I32Add])],
			},
			Block::Flat(vec![Instruction::I32Add]),
		];
		assert_eq!(expected[..], parse_blocks(source)[..]);
		assert_eq!(source[..], serialize(expected)[..]);
	}

	#[test]
	fn flat_blocks() {
		let mut blocks = vec![
			Block::Flat(vec![Instruction::I32Add]),
			Block::IfIns {
				ty: BlockType::NoResult,
				inner_true: vec![Block::Flat(vec![Instruction::I32Add])],
				inner_false: vec![Block::Flat(vec![Instruction::I32Add])],
			},
			Block::Flat(vec![Instruction::I32Add]),
		];
		let flat_blocks = flat_blocks_mut(&mut blocks);
		let flat_blocks_copy: Vec<Vec<Instruction>> =
			flat_blocks.iter().map(|x| (*x).clone()).collect();
		assert_eq!(
			vec![
				vec![Instruction::I32Add],
				vec![Instruction::I32Add],
				vec![Instruction::I32Add],
				vec![Instruction::I32Add],
			],
			flat_blocks_copy
		);
	}
}
