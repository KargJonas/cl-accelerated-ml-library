from typing import List, Set
from tensor import Tensor

bold = "\x1b[1m"
reset = "\x1b[0m"
purple = "\x1b[35m"
blue = "\x1b[1;34m"

def uid_to_color(uid: int, background: bool = False) -> str:
    uid += 1  # acts as a kind of seed
    r, g, b = (uid * 33457) % 256, (uid * 53457) % 256, (uid * 725099) % 256
    color_type = '48' if background else '38'
    return f"\033[{color_type};2;{r};{g};{b}m"

def get_tensor_name(tensor: Tensor):
    name = f"{tensor.op} | {tensor.shape} -> {tensor.strides} | {uid_to_color(tensor.uid)}{tensor.uid}{reset}"
    if tensor.name: name = f"{tensor.name} ({name})"
    return name

def tree_str(current_tensor: Tensor, visited: Set[Tensor] = set(), prefix: str = "", is_last: bool = True, is_root: bool = True) -> str:
    identifier = get_tensor_name(current_tensor)
    result = f" {bold}{identifier}{reset}\n" if is_root else f" {prefix}{'└─ ' if is_last else '├─ '}{identifier}\n"
    visited.add(current_tensor)
    new_prefix = prefix + f"{'   ' if is_last else '|  '}"
    for index, parent in enumerate(current_tensor.parents):
        parent_is_last = index == len(current_tensor.parents) - 1
        if id(parent) in visited:
            connector = "└─ " if parent_is_last else "├─ "
            parent_identifier = parent.op
            result += f"{new_prefix}{connector}{parent_identifier} (already visited){reset}\n"
        else:
            result += tree_str(parent, visited, new_prefix, parent_is_last, False)
    return result

def schedule_str(schedule: List[Tensor]):
    return "".join([f"{uid_to_color(tensor.uid, True)}  {tensor.uid}  {reset}" for tensor in schedule]) + "\n"
