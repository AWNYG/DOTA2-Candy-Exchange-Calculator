from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RULE_FILE = "ExchangeRule.txt"
DEFAULT_INVENTORY_FILE = "Inventory_Target.txt"


@dataclass(frozen=True)
class Step:
    label: str
    consume: tuple[int, ...]
    produce: tuple[int, ...]
    loss: int


def parse_counts(raw: str, expected_size: int | None = None) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.strip().split(";"))
    if expected_size is not None and len(values) != expected_size:
        raise ValueError(f"数据长度错误，期望 {expected_size} 个数字，实际得到 {len(values)} 个：{raw}")
    if any(value < 0 for value in values):
        raise ValueError(f"糖果数量不能为负数：{raw}")
    return values


def build_rule_step(label: str, consume: tuple[int, ...], produce: tuple[int, ...]) -> Step:
    return Step(label=label, consume=consume, produce=produce, loss=max(0, sum(consume) - sum(produce)))


def read_rules(path: Path) -> tuple[int, int, list[Step]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("ExchangeRule 文件格式错误，至少需要 CandyType、Rules、Maximum 三行头信息。")

    candy_type_prefix = "CandyType:"
    rules_prefix = "Rules:"
    maximum_prefix = "Maximum:"
    if not lines[0].startswith(candy_type_prefix):
        raise ValueError("ExchangeRule 第一行应为 CandyType:数字")
    if not lines[1].startswith(rules_prefix):
        raise ValueError("ExchangeRule 第二行应为 Rules:数字")
    if not lines[2].startswith(maximum_prefix):
        raise ValueError("ExchangeRule 第三行应为 Maximum:数字")

    candy_type_count = int(lines[0][len(candy_type_prefix) :].strip())
    rule_count = int(lines[1][len(rules_prefix) :].strip())
    maximum = int(lines[2][len(maximum_prefix) :].strip())
    if maximum <= 0:
        raise ValueError("Maximum 必须为正整数。")
    if len(lines) != rule_count + 3:
        raise ValueError(f"规则数量不匹配，声明 {rule_count} 条，实际读取到 {len(lines) - 3} 条。")

    rules: list[Step] = []
    for index, line in enumerate(lines[3:], start=1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"规则格式错误，第 {index} 条规则应为“消耗 TAB 获得”：{line}")
        consume = parse_counts(parts[0], candy_type_count)
        produce = parse_counts(parts[1], candy_type_count)
        rules.append(build_rule_step(str(index), consume, produce))
    return candy_type_count, maximum, rules


def read_inventory_and_target(path: Path, expected_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("Inventory_target 文件格式错误，应只有两行：库存和目标。")
    inventory = parse_counts(lines[0], expected_size)
    target = parse_counts(lines[1], expected_size)
    return inventory, target


def candy_names(count: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(count)]


def format_counts(values: Iterable[int], names: list[str]) -> str:
    return ", ".join(f"{name}={value}" for name, value in zip(names, values))


def total_candies(values: Iterable[int]) -> int:
    return sum(values)


def can_apply(state: tuple[int, ...], consume: tuple[int, ...]) -> bool:
    return all(current >= need for current, need in zip(state, consume))


def apply_step(state: tuple[int, ...], step: Step) -> tuple[int, ...]:
    return tuple(current - cost + gain for current, cost, gain in zip(state, step.consume, step.produce))


def is_goal(state: tuple[int, ...], target: tuple[int, ...]) -> bool:
    return all(current >= required for current, required in zip(state, target))


def build_builtin_lossy_steps(candy_count: int) -> list[Step]:
    steps: list[Step] = []
    for source in range(candy_count):
        for target in range(candy_count):
            if source == target:
                continue
            consume = [0] * candy_count
            produce = [0] * candy_count
            consume[source] = 3
            produce[target] = 1
            label = f"L({chr(ord('A') + source)}->{chr(ord('A') + target)})"
            steps.append(build_rule_step(label, tuple(consume), tuple(produce)))
    return steps


def reconstruct_path(
    end_state: tuple[int, ...],
    parent_map: dict[tuple[int, ...], tuple[tuple[int, ...] | None, Step | None]],
) -> list[tuple[tuple[int, ...], Step, tuple[int, ...]]]:
    path: list[tuple[tuple[int, ...], Step, tuple[int, ...]]] = []
    current = end_state
    while True:
        previous, step = parent_map[current]
        if previous is None or step is None:
            break
        path.append((previous, step, current))
        current = previous
    path.reverse()
    return path


def solve_max_growth(
    inventory: tuple[int, ...],
    maximum: int,
    steps: list[Step],
) -> tuple[list[tuple[tuple[int, ...], Step, tuple[int, ...]]] | None, int | None]:
    start_total = total_candies(inventory)
    parent_map: dict[tuple[int, ...], tuple[tuple[int, ...] | None, Step | None]] = {inventory: (None, None)}
    distance: dict[tuple[int, ...], int] = {inventory: 0}
    queue: list[tuple[int, ...]] = [inventory]
    head = 0

    best_state = inventory
    best_total = start_total
    best_steps = 0

    while head < len(queue):
        state = queue[head]
        head += 1
        current_steps = distance[state]

        for step in steps:
            if not can_apply(state, step.consume):
                continue
            next_state = apply_step(state, step)
            if total_candies(next_state) > maximum:
                continue
            if next_state in parent_map:
                continue

            parent_map[next_state] = (state, step)
            distance[next_state] = current_steps + 1
            queue.append(next_state)

            next_total = total_candies(next_state)
            next_steps = current_steps + 1
            if next_total > best_total or (next_total == best_total and next_steps < best_steps):
                best_state = next_state
                best_total = next_total
                best_steps = next_steps

    if best_total <= start_total:
        return None, None
    return reconstruct_path(best_state, parent_map), best_total


def solve_min_loss(
    inventory: tuple[int, ...],
    target: tuple[int, ...],
    maximum: int,
    steps: list[Step],
) -> tuple[list[tuple[tuple[int, ...], Step, tuple[int, ...]]] | None, int | None]:
    if is_goal(inventory, target):
        return [], 0

    parent_map: dict[tuple[int, ...], tuple[tuple[int, ...] | None, Step | None]] = {inventory: (None, None)}
    best_cost: dict[tuple[int, ...], tuple[int, int]] = {inventory: (0, 0)}
    heap: list[tuple[int, int, tuple[int, ...]]] = [(0, 0, inventory)]

    while heap:
        total_loss, total_steps, state = heapq.heappop(heap)
        if (total_loss, total_steps) != best_cost.get(state):
            continue
        if is_goal(state, target):
            return reconstruct_path(state, parent_map), total_loss

        for step in steps:
            if not can_apply(state, step.consume):
                continue
            next_state = apply_step(state, step)
            if total_candies(next_state) > maximum:
                continue
            next_cost = (total_loss + step.loss, total_steps + 1)
            if next_cost >= best_cost.get(next_state, (float("inf"), float("inf"))):
                continue
            best_cost[next_state] = next_cost
            parent_map[next_state] = (state, step)
            heapq.heappush(heap, (next_cost[0], next_cost[1], next_state))
    return None, None


def render_solution(
    inventory: tuple[int, ...],
    target: tuple[int, ...],
    maximum: int,
    names: list[str],
    path: list[tuple[tuple[int, ...], Step, tuple[int, ...]]] | None,
    total_loss: int | None,
) -> str:
    lines = [
        "最优方案：全局最小损耗",
        f"仓库容量上限: {maximum}",
        f"初始库存: {format_counts(inventory, names)} (总数={total_candies(inventory)})",
        f"目标需求: {format_counts(target, names)}",
    ]
    if path is None:
        lines.append("结果: 无解")
        return "\n".join(lines)

    if not path:
        lines.append("结果: 已满足目标，无需交换")
        lines.append("序列: (空)")
        lines.append(f"最终库存: {format_counts(inventory, names)} (总数={total_candies(inventory)})")
        lines.append("总损耗: 0")
        return "\n".join(lines)

    sequence = [step.label for _, step, _ in path]
    final_state = path[-1][2]
    lines.append("序列: " + " -> ".join(sequence))
    lines.append(f"步骤数: {len(path)}")
    lines.append(f"总损耗: {total_loss}")
    lines.append(f"最终库存: {format_counts(final_state, names)} (总数={total_candies(final_state)})")
    return "\n".join(lines)


def render_growth_solution(
    inventory: tuple[int, ...],
    maximum: int,
    names: list[str],
    path: list[tuple[tuple[int, ...], Step, tuple[int, ...]]] | None,
    best_total: int | None,
) -> str:
    start_total = total_candies(inventory)
    lines = [
        "库存增量分析：全部规则",
        f"仓库容量上限: {maximum}",
        f"初始库存: {format_counts(inventory, names)} (总数={start_total})",
    ]
    if path is None:
        lines.append("结果: 不能让总糖果数变多")
        return "\n".join(lines)

    sequence = [step.label for _, step, _ in path]
    final_state = path[-1][2]
    lines.append("结果: 可以让总糖果数变多")
    lines.append("序列: " + " -> ".join(sequence))
    lines.append(f"步骤数: {len(path)}")
    lines.append(f"最终库存: {format_counts(final_state, names)} (总数={total_candies(final_state)})")
    lines.append(f"净增量: {best_total - start_total}")
    return "\n".join(lines)


def validate_initial_state(inventory: tuple[int, ...], maximum: int) -> None:
    if total_candies(inventory) > maximum:
        raise ValueError(
            f"初始库存总数为 {total_candies(inventory)}，超过仓库容量上限 {maximum}。"
        )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rule_path = base_dir / DEFAULT_RULE_FILE
    inventory_path = base_dir / DEFAULT_INVENTORY_FILE

    candy_count, maximum, rule_steps = read_rules(rule_path)
    inventory, target = read_inventory_and_target(inventory_path, candy_count)
    validate_initial_state(inventory, maximum)

    all_steps = rule_steps + build_builtin_lossy_steps(candy_count)
    names = candy_names(candy_count)
    path, total_loss = solve_min_loss(inventory, target, maximum, all_steps)
    growth_path, best_total = solve_max_growth(inventory, maximum, all_steps)
    print(render_solution(inventory, target, maximum, names, path, total_loss))
    print()
    print(render_growth_solution(inventory, maximum, names, growth_path, best_total))


if __name__ == "__main__":
    main()
