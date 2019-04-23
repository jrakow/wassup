use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::iter::once;
use z3_sys2::*;

fn encode_init_conditions(ctx: Z3_context, solver: Z3_solver, program: &[Instruction]) {
	let word_sort = unsafe { Z3_mk_bv_sort(ctx, 32) };
	let int_sort = unsafe { Z3_mk_int_sort(ctx) };

	let stack_depth = 2;
	let stack_vars: Vec<_> = (0..stack_depth)
		.map(|i| unsafe {
			let name = Z3_mk_string_symbol(ctx, CString::new(format!("x{}", i)).unwrap().as_ptr());
			Z3_mk_const(ctx, name, word_sort)
		})
		.collect();

	// declare stack function
	let stack_func_name =
		unsafe { Z3_mk_string_symbol(ctx, CString::new("stack").unwrap().as_ptr()) };

	let stack_func_domain: Vec<_> = (0..stack_depth)
		.map(|_| word_sort) // stack variables
		.chain(once(int_sort)) // instruction counter
		.chain(once(int_sort)) // stack address
		.collect();
	let stack_func = unsafe {
		Z3_mk_func_decl(
			ctx,
			stack_func_name,
			stack_func_domain.len().try_into().unwrap(),
			stack_func_domain.as_ptr(),
			word_sort,
		)
	};

	// set stack(xs, 0, i) == xs[i]
	let program_counter = unsafe { Z3_mk_int64(ctx, 0, int_sort) };
	for i in 0..stack_depth {
		let stack_index = unsafe { Z3_mk_int(ctx, i as _, int_sort) };
		let args: Vec<_> = stack_vars
			.iter()
			.cloned()
			.chain(once(program_counter))
			.chain(once(stack_index))
			.collect();
		let lhs = unsafe { Z3_mk_app(ctx, stack_func, args.len() as _, args.as_ptr()) };
		let rhs = stack_vars[i];
		unsafe { Z3_solver_assert(ctx, solver, Z3_mk_eq(ctx, lhs, rhs)); }
	};

	// declare stack pointer function
	let stack_pointer_name =
		unsafe { Z3_mk_string_symbol(ctx, CString::new("stack_pointer").unwrap().as_ptr()) };
	let stack_pointer_func_domain = &[int_sort];
	let stack_pointer_func = unsafe {
		Z3_mk_func_decl(
			ctx,
			stack_pointer_name,
			stack_pointer_func_domain.len() as _,
			stack_pointer_func_domain.as_ptr(),
			int_sort,
		)
	};

	// set stack_counter(0) = 0
	let stack_pointer_init_condition = unsafe {
		let zero = Z3_mk_int(ctx, 0, int_sort);
		let args = &[zero];
		Z3_solver_assert(ctx, solver, Z3_mk_eq(
			ctx,
			Z3_mk_app(ctx, stack_pointer_func, args.len() as _, args.as_ptr()),
			zero,
		))
	};
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn simple() {
		let ctx = unsafe { Z3_mk_context(Z3_mk_config()) };
		let solver = unsafe { Z3_mk_solver(ctx) };
		encode_init_conditions(
			ctx,
			solver,
			&[
				Instruction::I32Const(1),
				Instruction::I32Const(2),
				Instruction::I32Add,
			],
		);
		unsafe { Z3_solver_check(ctx, solver) };
		let model = unsafe { Z3_solver_get_model(ctx, solver) };
		let model_string = unsafe { &CStr::from_ptr(Z3_model_to_string(ctx, model)) };
		println!("{}", model_string.to_str().unwrap());
	}
}
