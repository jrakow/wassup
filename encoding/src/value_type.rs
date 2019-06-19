use z3::*;

/// Encoding of a Wasm value type: I32, I64, F32, F64
pub fn value_type_sort(ctx: &Context) -> (Sort, Vec<Ast>, Vec<FuncDecl>) {
	let (sort, consts, testers) = ctx.enumeration_sort(
		&ctx.str_sym("ValueType"),
		&[&ctx.str_sym("I32"), &ctx.str_sym("I64")],
	);
	let consts = consts.iter().map(|c| c.apply(&[])).collect();
	(sort, consts, testers)
}
