use crate::block::flat_blocks_mut;
use parity_wasm::elements::Instruction;

mod block;
mod encoding;

pub use parity_wasm;

pub fn superoptimize_instructions(source_program: &[Instruction]) -> Vec<Instruction> {
	encoding::superoptimize(source_program)
}

pub fn superoptimize_func_body(func_body: &mut parity_wasm::elements::FuncBody) {
	let code = func_body.code_mut().elements_mut();
	let mut blocks = block::parse_blocks(code);
	let flat_blocks = flat_blocks_mut(&mut blocks);
	for flat_block in flat_blocks {
		let instructions = &flat_block[..];
		*flat_block = superoptimize_instructions(instructions);
	}
	*code = block::serialize(&blocks);
}

pub fn superoptimize_module(module: &mut parity_wasm::elements::Module) {
	if let Some(code_section) = module.code_section_mut() {
		for func_body in code_section.bodies_mut() {
			superoptimize_func_body(func_body);
		}
	}
}
