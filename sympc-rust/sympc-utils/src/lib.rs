use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(FromPyObject)]
struct RustObjPointer {
    __name__: String,
}

#[pyfunction]
fn ispointer(py:Python, obj: PyObject) -> PyResult<bool> {
    let rustObjPointer: Result<RustObjPointer, PyErr> = obj.extract(py)?;
    Ok(false)
}


#[pyfunction]
fn islocal(obj: PyObject) -> PyResult<bool> {
    Ok(true)
}

#[pymodule]
fn sympc_utils(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(islocal, m)?)?;
    m.add_function(wrap_pyfunction!(ispointer, m)?)?;

    Ok(())
}
