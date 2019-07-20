use parity_wasm::elements::{Internal, Module, Type, ValueType};
use std::{collections::HashMap, fs::read};
use wabt::script::{Action, Action::*, Command, CommandKind, ScriptParser, Value as ScriptValue};
use wassup_encoding::*;
use z3::*;

#[test]
fn test_i32() {
	spec_test("i32.wast")
}

#[test]
fn test_i64() {
	spec_test("i64.wast")
}

#[test]
fn test_select() {
	spec_test("select.wast")
}

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

			let instr = body.code().elements();
			// slice of last End
			let instr = &instr[..instr.len() - 1];

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
				initial_locals,
				function.local_types.clone(),
				&[],
				value_type_config,
			);
			let source_program = function.instructions[0].as_ref().left().unwrap();
			let state = State::new(&ctx, &solver, &constants, "", source_program.len());

			state.set_source_program(source_program);
			solver.assert(&state.transitions());

			assert!(solver.check());
			let model = solver.get_model();

			let pc = ctx.from_usize(instr.len());
			let sp = model
				.eval(&state.stack_pointer(&pc))
				.unwrap()
				.as_usize()
				.unwrap();
			if sp > 0 {
				let sp = state.stack_pointer(&pc).sub(&[&ctx.from_u64(1)]);
				let i = Value::decode(
					&state.stack(&pc, &sp),
					&model,
					value_type_config.decode_value_type(&ctx, &model, &state.stack_type(&pc, &sp)),
					value_type_config,
				);
				Some(vec![i])
			} else {
				None
			}
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
