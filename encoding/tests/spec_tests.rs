use either::Either;
use parity_wasm::elements::{Internal, Module, Type, ValueType};
use std::{collections::HashMap, fs::read};
use wabt::script::{
	Action::{self, *},
	Command, CommandKind, ScriptParser, Value as ScriptValue,
};
use wassup_encoding::*;
use z3::*;

#[test]
fn test_address() { spec_test("address.wast") }

#[test]
fn test_align() { spec_test("align.wast") }

#[test]
fn test_binary_leb128() { spec_test("binary-leb128.wast") }

#[test]
fn test_block() { spec_test("block.wast") }

#[test]
fn test_break_drop() { spec_test("break-drop.wast") }

#[test]
fn test_br_if() { spec_test("br_if.wast") }

#[test]
fn test_br_table() { spec_test("br_table.wast") }

#[test]
fn test_br() { spec_test("br.wast") }

#[test]
fn test_call_indirect() { spec_test("call_indirect.wast") }

#[test]
fn test_call() { spec_test("call.wast") }

#[test]
fn test_comments() { spec_test("comments.wast") }

#[test]
fn test_conversions() { spec_test("conversions.wast") }

#[test]
fn test_custom() { spec_test("custom.wast") }

#[test]
fn test_data() { spec_test("data.wast") }

#[test]
fn test_elem() { spec_test("elem.wast") }

#[test]
fn test_endianness() { spec_test("endianness.wast") }

#[test]
fn test_f32_bitwise() { spec_test("f32_bitwise.wast") }

#[test]
fn test_f32_cmp() { spec_test("f32_cmp.wast") }

#[test]
fn test_f32() { spec_test("f32.wast") }

#[test]
fn test_f64_bitwise() { spec_test("f64_bitwise.wast") }

#[test]
fn test_f64_cmp() { spec_test("f64_cmp.wast") }

#[test]
fn test_f64() { spec_test("f64.wast") }

#[test]
fn test_fac() { spec_test("fac.wast") }

#[test]
fn test_float_exprs() { spec_test("float_exprs.wast") }

#[test]
fn test_float_literals() { spec_test("float_literals.wast") }

#[test]
fn test_float_memory() { spec_test("float_memory.wast") }

#[test]
fn test_float_misc() { spec_test("float_misc.wast") }

#[test]
fn test_forward() { spec_test("forward.wast") }

#[test]
#[ignore]
fn test_func_ptrs() { spec_test("func_ptrs.wast") }

#[test]
#[ignore]
fn test_func() { spec_test("func.wast") }

#[test]
fn test_globals() { spec_test("globals.wast") }

#[test]
fn test_i32() { spec_test("i32.wast") }

#[test]
fn test_i64() { spec_test("i64.wast") }

#[test]
fn test_if() { spec_test("if.wast") }

#[test]
fn test_inline_module() { spec_test("inline-module.wast") }

#[test]
fn test_int_exprs() { spec_test("int_exprs.wast") }

#[test]
fn test_int_literals() { spec_test("int_literals.wast") }

#[test]
fn test_labels() { spec_test("labels.wast") }

#[test]
fn test_left_to_right() { spec_test("left-to-right.wast") }

#[test]
fn test_load() { spec_test("load.wast") }

#[test]
fn test_local_get() { spec_test("local_get.wast") }

#[test]
fn test_local_set() { spec_test("local_set.wast") }

#[test]
fn test_local_tee() { spec_test("local_tee.wast") }

#[test]
fn test_loop() { spec_test("loop.wast") }

#[test]
fn test_memory_grow() { spec_test("memory_grow.wast") }

#[test]
fn test_memory_redundancy() { spec_test("memory_redundancy.wast") }

#[test]
fn test_memory_size() { spec_test("memory_size.wast") }

#[test]
fn test_memory_trap() { spec_test("memory_trap.wast") }

#[test]
fn test_memory() { spec_test("memory.wast") }

#[test]
fn test_nop() { spec_test("nop.wast") }

#[test]
fn test_return() { spec_test("return.wast") }

#[test]
fn test_select() { spec_test("select.wast") }

#[test]
fn test_skip_stack_guard_page() { spec_test("skip-stack-guard-page.wast") }

#[test]
fn test_stack() { spec_test("stack.wast") }

#[test]
fn test_start() { spec_test("start.wast") }

#[test]
fn test_store() { spec_test("store.wast") }

#[test]
fn test_switch() { spec_test("switch.wast") }

#[test]
fn test_token() { spec_test("token.wast") }

#[test]
fn test_traps() { spec_test("traps.wast") }

#[test]
fn test_typecheck() { spec_test("typecheck.wast") }

#[test]
fn test_type() { spec_test("type.wast") }

#[test]
fn test_unreachable() { spec_test("unreachable.wast") }

#[test]
fn test_unreached_invalid() { spec_test("unreached-invalid.wast") }

#[test]
fn test_unwind() { spec_test("unwind.wast") }

#[test]
fn test_utf8_custom_section_id() { spec_test("utf8-custom-section-id.wast") }

#[test]
fn test_utf8_import_field() { spec_test("utf8-import-field.wast") }

#[test]
fn test_utf8_import_module() { spec_test("utf8-import-module.wast") }

#[test]
fn test_utf8_invalid_encoding() { spec_test("utf8-invalid-encoding.wast") }

fn spec_test(name: &str) {
	let source = read("tests/spec_tests/".to_owned() + name).unwrap();

	let mut parser: ScriptParser<f32, f64> =
		ScriptParser::from_source_and_name(&source, name).unwrap();

	let mut modules = HashMap::new();
	while let Some(Command { kind, .. }) = parser.next().unwrap() {
		match kind {
			CommandKind::Module {
				module: binary,
				name,
			} => {
				let module = parity_wasm::deserialize_buffer::<Module>(&binary.into_vec()).unwrap();
				modules.insert(name, module);
			}
			CommandKind::AssertReturn { action, expected } => {
				// ignore unencodable values
				let expected: Vec<_> = expected.iter().cloned().map(value_cast).collect();
				if expected.iter().any(Option::is_none) {
					continue;
				}
				let expected: Vec<_> = expected.into_iter().map(Option::unwrap).collect();

				// ignore unencodable functions
				if let Some(res) = action_result(&modules, action) {
					assert_eq!(res, expected);
				}
			}
			CommandKind::AssertTrap { .. } => {} // TODO
			_ => {}
		}
	}
}

fn action_result(
	modules: &HashMap<Option<String>, Module>,
	action: Action<f32, f64>,
) -> Option<Vec<Value>> {
	match action {
		Invoke {
			module: module_name,
			field,
			args,
		} => {
			// Instead of invoking the function, encode it and the starting values.
			// Then get a model and grab the result.

			let module = &modules[&module_name];
			let func_index = get_function_index(module, &field);
			let func = module.function_section().unwrap().entries()[func_index];
			let params = match &module.type_section().unwrap().types()[func.type_ref() as usize] {
				Type::Function(f) => f.params(),
			};
			let body = &module.code_section().unwrap().bodies()[func_index];
			let function = Function::from_wasm_func_body_params(body, params);

			// bail on functions with unencodable instructions
			if function.instructions.len() > 2 || function.instructions[0].is_right() {
				return None;
			}

			let ctx = Context::new(&Config::default());
			let solver = Solver::new(&ctx);
			let value_type_config = ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			};

			// create values of initial locals: arguments then 0s for all other locals
			let args: Vec<_> = args
				.iter()
				.cloned()
				.map(value_cast)
				.map(Option::unwrap)
				.collect();
			let remaining_locals: Vec<_> = function.local_types[function.n_params..]
				.iter()
				.map(|ty| match ty {
					ValueType::I32 => Value::I32(0),
					ValueType::I64 => Value::I64(0),
					_ => unimplemented!(),
				})
				.collect();
			let initial_locals = args
				.iter()
				.chain(remaining_locals.iter())
				.map(|v| v.encode(&ctx, value_type_config))
				.collect();

			let constants = Constants::new(
				&ctx,
				&solver,
				vec![],
				initial_locals,
				function.local_types.clone(),
				&[],
				value_type_config,
			);
			let source_program = function.instructions[0].as_ref().left().unwrap();
			let execution = Execution::new(
				&constants,
				&solver,
				"".to_owned(),
				Either::Left(source_program),
			);

			assert!(solver.check().unwrap());
			let model = solver.get_model();

			Some(
				execution.states[source_program.len()]
					.decode(&model, &constants)
					.stack,
			)
		}
		// TODO trapped tests
		_ => panic!(),
	}
}

fn get_function_index(module: &Module, name: &str) -> usize {
	let exports = module.export_section().unwrap().entries();
	for export in exports {
		if export.field() == name {
			match export.internal() {
				Internal::Function(i) => {
					return *i as usize;
				}
				_ => panic!("export is not a function"),
			}
		}
	}
	panic!("function not found");
}

fn value_cast(v: ScriptValue<f32, f64>) -> Option<Value> {
	match v {
		ScriptValue::I32(i) => Some(wassup_encoding::Value::I32(i)),
		ScriptValue::I64(i) => Some(wassup_encoding::Value::I64(i)),
		_ => None,
	}
}
