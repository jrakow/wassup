use parity_wasm::elements::{Instruction, Module};
use wassup_encoding::ValueTypeConfig;

#[test]
fn module_consts_add() {
	let wasm_binary = wabt::Wat2Wasm::new()
		.convert(
			r#"(module
				(func (result i32)
					i32.const 1
					nop
					i32.const 2
					i32.add
				)
			)"#,
		)
		.unwrap();
	let mut module: Module = parity_wasm::deserialize_buffer(wasm_binary.as_ref()).unwrap();

	wassup::superoptimize_module(
		&mut module,
		ValueTypeConfig {
			i32_size: 4,
			i64_size: Some(8),
		},
		Some(ValueTypeConfig {
			i32_size: 4,
			i64_size: Some(8),
		}),
		None,
	);

	let func_body = &module.code_section().unwrap().bodies()[0];
	assert_eq!(
		func_body.code().elements(),
		&[Instruction::I32Const(3), Instruction::End]
	);
	assert!(func_body.locals().is_empty());
}
