use crate::block::flat_blocks_mut;
use parity_wasm::elements::{
	CodeSection, FunctionSection, Instruction, Type, TypeSection, ValueType,
};
use rayon::prelude::*;

mod block;
mod encoding;

pub use parity_wasm;

pub fn superoptimize_instructions(
	source_program: &[Instruction],
	n_params: usize,
) -> Vec<Instruction> {
	encoding::superoptimize(source_program, n_params)
}

pub fn superoptimize_func_body(
	func_body: &mut parity_wasm::elements::FuncBody,
	params: &[ValueType],
) {
	let n_params = params.len();

	let code = func_body.code_mut().elements_mut();
	let mut blocks = block::parse_blocks(code);
	let flat_blocks = flat_blocks_mut(&mut blocks);
	for flat_block in flat_blocks {
		let instructions = &flat_block[..];
		*flat_block = superoptimize_instructions(instructions, n_params);
	}
	*code = block::serialize(&blocks);
}

pub fn superoptimize_module(module: &mut parity_wasm::elements::Module) {
	// Destructure the module
	// Safe, because the sections should be independent
	let (type_section, code_section, function_section): (
		&TypeSection,
		&mut CodeSection,
		&FunctionSection,
	) = unsafe {
		(
			&*(module.type_section().unwrap() as *const _),
			&mut *(module.code_section_mut().unwrap() as *mut _),
			&*(module.function_section().unwrap() as *const _),
		)
	};

	code_section
		.bodies_mut()
		.par_iter_mut()
		.enumerate()
		.for_each(|(index, body)| {
			let func = function_section.entries()[index];
			let signature = &type_section.types()[func.type_ref() as usize];
			let params = match signature {
				Type::Function(f) => f.params(),
			};
			superoptimize_func_body(body, params)
		});
}
