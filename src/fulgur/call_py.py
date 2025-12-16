import cloudpickle as cp
import multiprocessing as mp
import polars as pl
from tqdm import tqdm
from typing import Any, Callable, List

from fulgur.utils import nrow


def _worker(fn_bytes: bytes, args_bytes: bytes, return_queue: mp.Queue):
    try:
        fn = cp.loads(fn_bytes)
        args, kwargs = cp.loads(args_bytes)
        result = fn(*args, **kwargs)
        return_queue.put(("ok", cp.dumps(result)))
    except Exception as e:
        return_queue.put(("error", cp.dumps(e)))


def call_py(function: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Execute a function in a completely isolated subprocess and return the result.
    Supports lambdas, closures, functions using external packages, etc.
    """
    queue = mp.Queue()

    fn_bytes = cp.dumps(function)
    args_bytes = cp.dumps((args, kwargs))

    # Start sub-process and execute function; then close/cleanup process & queue
    p = mp.Process(target=_worker, args=(fn_bytes, args_bytes, queue))
    p.start()
    status, result_bytes = queue.get()
    p.join()
    p.close()
    queue.close()
    queue.join_thread()

    # Handle errors
    if status == "error":
        raise cp.loads(result_bytes)

    return cp.loads(result_bytes)


# Stream data from a Polars LazyFrame object and apply a function to all chunks
# and return the result. You may ask, why on earth execut this in a sub-process??
# The answer is because Polars is, in this case, stupid (sadly)! Basically, it
# does not do a good job of freeing up memory that it has allocated after the process
# is done. The best way to force this to happen is to run it in a sub-process which
# closes after the function returns. See the following related GitHub issues:
# https://github.com/pola-rs/polars/issues/23128
# https://github.com/pola-rs/polars/issues/22871
# https://github.com/pola-rs/polars/issues/21497#issuecomment-2688338337
# TODO: This still accumulates memory in the sub-process. Is there a better/more efficient
# way to be doing this? Could run each chunk in its own sub-process but that makes it sooooo
# much slower...
def stream_data(
    data: pl.LazyFrame,
    fn: Callable[[pl.DataFrame], Any],  # Function to apply to each chunk
    query: Callable[[pl.LazyFrame], Any] | None = None,  # Additional querying to do
    batch_size: int = 1000,
    last: bool = False,  # Return only the last result (as opposed to all results)
    verbose: bool = True,  # Print progress
) -> List[Any] | None:
    """
    Apply an arbitrary function to a stream of data and collect the results (if any).
    """
    if not isinstance(data, pl.LazyFrame):
        raise ValueError("`data` must be a Polars LazyFrame object")
    if query:
        data = query(data)
    # Running with n_chunks may add unnecessary overhead; TODO: test on massive (billions?) dataset
    nrow_data = nrow(data)
    # Calculate the total number of chunks to be processed
    if (nrow_data % batch_size) == 0:
        n_chunks = nrow_data // batch_size
    else:
        n_chunks = (nrow(data) // batch_size) + 1
    # Create a (potentially tqdm-wrapped) iterator
    if verbose:
        chunks = tqdm(data.collect_batches(chunk_size=batch_size), total=n_chunks)
    else:
        chunks = data.collect_batches(chunk_size=batch_size)
    # Collect results of function applied to full dataset
    out = list()
    for chunk in chunks:
        result = fn(chunk)
        if result is not None and not last:
            out.append(result)
        continue
    if len(out) > 0:
        return out
    elif last:
        return result
    else:
        return None
