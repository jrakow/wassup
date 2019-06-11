use super::*;
use parity_wasm::elements::{Internal, Module, Type};
use std::{collections::HashMap, fs::read};
use wabt::script::{Action, Action::*, Command, CommandKind, ScriptParser, Value};

#[test]
#[ignore]
fn test_i32() {
	spec_test("i32.wast")
}

fn spec_test(name: &str) {
	let source = read("src/encoding/spec_tests/".to_owned() + name).unwrap();

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
				let res = action_result(&modules, action);
				assert_eq!(res, value_cast(expected[0]));
			}
			CommandKind::AssertTrap { .. } => {} // TODO
			_ => {}
		}
	}
}

fn action_result(modules: &HashMap<Option<String>, Module>, action: Action<f32, f64>) -> u32 {
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

			let instr = body.code().elements();
			// slice of last End
			let instr = &instr[..instr.len() - 1];

			let ctx = Context::new(&Config::default());
			let solver = Solver::new(&ctx);
			let constants =
				Constants::new(&ctx, &solver, stack_depth(instr) as usize, params.len());
			let state = State::new(&ctx, &solver, &constants, "");

			// set initial values
			let args: Vec<_> = args.iter().cloned().map(value_cast).collect();;
			constants.set_params(&solver, &args);
			state.set_source_program(instr);
			state.assert_transitions();

			assert!(solver.check());
			let model = solver.get_model();

			let pc = ctx.from_u64(instr.len() as u64);
			let i = model
				.eval(
					&state
						.stack(&pc, &state.stack_pointer(&pc).sub(&[&ctx.from_u64(1)]))
						.bv2int(false),
				)
				.unwrap()
				.as_i64()
				.unwrap() as u32;
			i
		}
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

fn value_cast(v: Value<f32, f64>) -> u32 {
	match v {
		Value::I32(i) => i as _,
		_ => unimplemented!(),
	}
}
