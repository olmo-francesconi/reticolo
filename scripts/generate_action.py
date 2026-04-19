#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

VALID_ALGORITHMS = ("Metropolis", "HMC", "LLRMetropolis")
CUSTOMIZABLE_ALGORITHMS = ("Metropolis", "HMC")
FIELD_TYPE_CHOICES = (
    ("complex_scalar", "Complex scalar field", "std::complex<TImpl>"),
    ("metric_hfield", "Weak-field metric tensor field", "HField<TImpl>"),
)
ACTION_TYPE_CHOICES = (
    ("real_scalar", "Real-valued action", "TImpl"),
    ("complex_scalar", "Complex-valued action", "std::complex<TImpl>"),
)
OBSERVABLE_TYPE_CHOICES = (
    ("impl_type", "Template precision scalar", "impl_type"),
    ("float", "Explicit float", "float"),
    ("double", "Explicit double", "double"),
    ("int", "Signed integer", "int"),
    ("unsigned", "Unsigned integer", "unsigned"),
)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    action_dir: Path
    adapter_dir: Path
    registration_dir: Path
    storage_dir: Path
    tests_dir: Path
    scripts_input_dir: Path
    action_descriptor: Path
    builtin_action_families: Path
    builtin_module_registration: Path
    builtin_mc_adapters: Path
    hdf5_type_mappings: Path
    registry_metadata_test: Path


class PatchError(RuntimeError):
    pass


def project_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        action_dir=root / "src/reticolo/action",
        adapter_dir=root / "src/reticolo/action/adapters",
        registration_dir=root / "src/reticolo/action/registration",
        storage_dir=root / "src/reticolo/action/storage",
        tests_dir=root / "tests",
        scripts_input_dir=root / "scripts/input",
        action_descriptor=root / "src/reticolo/action/registration/ActionDescriptor.hpp",
        builtin_action_families=root / "src/reticolo/action/registration/BuiltinActionFamilies.def",
        builtin_module_registration=root / "src/reticolo/modules/factory/BuiltinModuleRegistration.hpp",
        builtin_mc_adapters=root / "src/reticolo/action/adapters/BuiltinMonteCarloAdapters.hpp",
        hdf5_type_mappings=root / "src/reticolo/core/storage/Hdf5TypeMappings.hpp",
        registry_metadata_test=root / "tests/registry_metadata.cpp",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold a new reticolo action family and patch the registration files."
    )
    parser.add_argument("name", nargs="?", help="Action class name, for example MyNewAction")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive terminal wizard instead of relying only on CLI arguments",
    )
    parser.add_argument(
        "--field-type",
        default="std::complex<TImpl>",
        help="Field type used by the action template. Default: std::complex<TImpl>",
    )
    parser.add_argument(
        "--action-type",
        default="std::complex<TImpl>",
        help="Action return type. Default: std::complex<TImpl>",
    )
    parser.add_argument(
        "--module-name",
        default="MonteCarlo",
        help="Runtime module name. Default: MonteCarlo",
    )
    parser.add_argument(
        "--storage-schema",
        default=None,
        help="Storage schema identifier. Default: snake_case(name)",
    )
    parser.add_argument(
        "--algorithms",
        default="Metropolis",
        help="Comma-separated algorithms from Metropolis,HMC,LLRMetropolis",
    )
    parser.add_argument(
        "--custom-algorithms",
        default="",
        help="Comma-separated subset of Metropolis,HMC for custom adapter stubs",
    )
    parser.add_argument(
        "--observable-fields",
        default="placeholder:impl_type",
        help="Comma-separated observable fields in name:type form. Default: placeholder:impl_type",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files and patches that would be applied without writing them",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root. Default: repository root inferred from this script",
    )
    return parser.parse_args()


def ensure_action_name(name: str) -> None:
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
        raise SystemExit("action name must be PascalCase and start with an uppercase letter")


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped


def parse_observables(value: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for raw_field in parse_csv(value):
        parts = [part.strip() for part in raw_field.split(":", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise SystemExit(f"invalid observable field '{raw_field}', expected name:type")
        fields.append((parts[0], parts[1]))
    return fields


def prompt_with_default(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        response = input(f"{prompt}{suffix}: ").strip()
        if response:
            return response
        if default is not None:
            return default
        print("A value is required.")


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_hint}]: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_menu(
    title: str,
    choices: tuple[tuple[str, str, str], ...],
    default_value: str | None = None,
) -> str:
    default_index = None
    for index, (_, _, value) in enumerate(choices, start=1):
        if value == default_value:
            default_index = index
            break

    while True:
        print(title)
        for index, (_, label, value) in enumerate(choices, start=1):
            print(f"  {index}. {label} -> {value}")
        suffix = f" [{default_index}]" if default_index is not None else ""
        raw = input(f"Select an option by number{suffix}: ").strip()
        if not raw and default_index is not None:
            return choices[default_index - 1][2]
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1][2]
        print("Please enter one of the listed numbers.")


def prompt_choice_list(
    prompt: str,
    allowed: tuple[str, ...],
    default: list[str] | None = None,
    allow_empty: bool = False,
) -> list[str]:
    default_text = ",".join(default) if default else None
    allowed_text = ", ".join(allowed)
    while True:
        response = prompt_with_default(f"{prompt}\nAllowed values: {allowed_text}", default_text)
        values = parse_csv(response)
        if not values and allow_empty:
            return []
        invalid = [value for value in values if value not in allowed]
        if invalid:
            print("Unsupported value(s): " + ", ".join(invalid))
            continue
        return values


def prompt_multi_select_menu(
    title: str,
    allowed: tuple[str, ...],
    default: list[str] | None = None,
    allow_empty: bool = False,
) -> list[str]:
    default = default or []
    default_indices = [str(index) for index, value in enumerate(allowed, start=1) if value in default]
    default_text = ",".join(default_indices) if default_indices else None
    while True:
        print(title)
        for index, value in enumerate(allowed, start=1):
            print(f"  {index}. {value}")
        raw = prompt_with_default(
            "Select one or more options by number, separated by commas",
            default_text,
        )
        values: list[str] = []
        ok = True
        for token in [part.strip() for part in raw.split(",") if part.strip()]:
            if not token.isdigit():
                ok = False
                break
            index = int(token)
            if index < 1 or index > len(allowed):
                ok = False
                break
            value = allowed[index - 1]
            if value not in values:
                values.append(value)
        if not ok:
            print("Please enter only numbers from the list, separated by commas.")
            continue
        if not values and not allow_empty:
            print("Please select at least one option.")
            continue
        return values


def prompt_observables(default: str) -> str:
    explanation = (
        "Observable fields define what gets written to the measurements dataset.\n"
        "Use comma-separated name:type entries. Primitive types only for automatic HDF5 mapping.\n"
        "Examples: energy:impl_type, density:double, accepted:int"
    )
    while True:
        response = prompt_with_default(explanation, default)
        try:
            parse_observables(response)
        except SystemExit as exc:
            print(exc)
            continue
        return response


def ensure_identifier(name: str, what: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise SystemExit(f"{what} must be a valid C++/YAML identifier using letters, digits, and underscores")


def prompt_observable_entries(existing: str) -> str:
    current = parse_observables(existing)
    while True:
        print(
            "Observable fields define the struct written to the measurements dataset.\n"
            "You will enter them one at a time. Names become struct fields and HDF5 column names."
        )
        default_count = max(1, len(current))
        while True:
            count_raw = prompt_with_default(
                "How many observable fields should be generated?",
                str(default_count),
            )
            if count_raw.isdigit() and int(count_raw) > 0:
                count = int(count_raw)
                break
            print("Please enter a positive integer.")

        fields: list[tuple[str, str]] = []
        for index in range(count):
            previous_name = current[index][0] if index < len(current) else None
            while True:
                field_name = prompt_with_default(
                    f"Observable {index + 1} name",
                    previous_name,
                )
                try:
                    ensure_identifier(field_name, f"observable {index + 1} name")
                except SystemExit as exc:
                    print(exc)
                    continue
                if any(existing_name == field_name for existing_name, _ in fields):
                    print("Observable names must be unique.")
                    continue
                break

            previous_type = current[index][1] if index < len(current) else "impl_type"
            field_type = prompt_menu(
                f"Observable {index + 1} type.\n"
                "Choose one of the built-in primitive mappings supported by the automatic HDF5 generator.",
                OBSERVABLE_TYPE_CHOICES,
                default_value=previous_type,
            )
            fields.append((field_name, field_type))

        print("Observable summary:")
        for field_name, field_type in fields:
            print(f"  - {field_name}: {field_type}")
        if prompt_yes_no("Use this observable list?", default=True):
            return ",".join(f"{field_name}:{field_type}" for field_name, field_type in fields)


def run_interactive_wizard(args: argparse.Namespace) -> argparse.Namespace:
    print("Reticolo action scaffold wizard")
    print()
    print("This will generate a new action family, patch the builtin registry files,")
    print("and leave IMPLEMENTATION REQUIRED blocks where the physics-specific code belongs.")
    print()

    while True:
        name = prompt_with_default(
            "Action class name.\nUse PascalCase because the generator derives filenames, descriptor names, and YAML action names from it",
            args.name,
        )
        try:
            ensure_action_name(name)
            args.name = name
            break
        except SystemExit as exc:
            print(exc)

    args.field_type = prompt_menu(
        "Field type template.\nChoose from field types already present in the codebase.",
        FIELD_TYPE_CHOICES,
        default_value=args.field_type,
    )
    args.action_type = prompt_menu(
        "Action return type.\nChoose the value category returned by compute_S and related methods.",
        ACTION_TYPE_CHOICES,
        default_value=args.action_type,
    )
    args.module_name = prompt_with_default(
        "Runtime module name.\nLeave this as MonteCarlo unless you are wiring the action into a different module family",
        args.module_name,
    )

    default_schema = args.storage_schema or snake_case(args.name)
    args.storage_schema = prompt_with_default(
        "Storage schema identifier.\nThis tags the action family in runtime metadata and on-disk schema naming",
        default_schema,
    )

    args.algorithms = ",".join(
        prompt_multi_select_menu(
            "Supported builtin algorithm names.\nPick the algorithms that should be exposed in metadata and accepted by the runtime.",
            VALID_ALGORITHMS,
            default=parse_csv(args.algorithms) or ["Metropolis"],
        )
    )

    wants_custom = prompt_yes_no(
        "Do you want custom Monte Carlo adapter stubs?\nChoose yes if the generic Metropolis/HMC implementations are not sufficient for this action",
        default=bool(parse_csv(args.custom_algorithms)),
    )
    if wants_custom:
        available_custom = tuple(
            algorithm for algorithm in parse_csv(args.algorithms) if algorithm in CUSTOMIZABLE_ALGORITHMS
        )
        if not available_custom:
            print("No custom adapter targets are available because only Metropolis and HMC support stub generation.")
            args.custom_algorithms = ""
        else:
            args.custom_algorithms = ",".join(
                prompt_multi_select_menu(
                    "Which custom adapter stubs should be generated?\nOnly select algorithms you plan to override manually.",
                    available_custom,
                    default=[available_custom[0]],
                    allow_empty=False,
                )
            )
    else:
        args.custom_algorithms = ""

    args.observable_fields = prompt_observable_entries(args.observable_fields)

    args.force = prompt_yes_no(
        "Overwrite generated files if they already exist?\nLeave this off unless you intentionally want to replace a previous scaffold",
        default=args.force,
    )
    args.dry_run = prompt_yes_no(
        "Start with a preview instead of writing immediately?",
        default=args.dry_run,
    )

    while True:
        print()
        print("Review")
        print(f"  Action name: {args.name}")
        print(f"  Field type: {args.field_type}")
        print(f"  Action type: {args.action_type}")
        print(f"  Module name: {args.module_name}")
        print(f"  Storage schema: {args.storage_schema}")
        print(f"  Algorithms: {args.algorithms}")
        print(f"  Custom algorithms: {args.custom_algorithms or '(none)'}")
        print(f"  Observable fields: {args.observable_fields}")
        print(f"  Overwrite existing files: {'yes' if args.force else 'no'}")
        print(f"  Initial mode: {'dry-run preview' if args.dry_run else 'write files'}")
        print()
        print("  1. Proceed")
        print("  2. Toggle preview/write mode")
        print("  3. Restart wizard")
        print("  4. Exit")
        choice = input("Select an option [1]: ").strip() or "1"
        if choice == "1":
            return args
        if choice == "2":
            args.dry_run = not args.dry_run
            continue
        if choice == "3":
            print()
            return run_interactive_wizard(
                argparse.Namespace(
                    name=args.name,
                    interactive=True,
                    field_type=args.field_type,
                    action_type=args.action_type,
                    module_name=args.module_name,
                    storage_schema=args.storage_schema,
                    algorithms=args.algorithms,
                    custom_algorithms=args.custom_algorithms,
                    observable_fields=args.observable_fields,
                    force=args.force,
                    dry_run=args.dry_run,
                    root=args.root,
                )
            )
        if choice == "4":
            raise SystemExit(0)
        print("Please enter 1, 2, 3, or 4.")


def snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def descriptor_name(name: str) -> str:
    return f"{name}Descriptor"


def registration_function_name(name: str) -> str:
    return f"register_{snake_case(name)}_modules"


def bool_literal(value: bool) -> str:
    return "true" if value else "false"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, contents: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text(contents, encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


def create_file(path: Path, contents: str, *, force: bool, dry_run: bool) -> str:
    if path.exists() and not force:
        raise PatchError(f"refusing to overwrite existing file: {path}")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return f"create {display_path(path)}"


def ensure_line(text: str, line: str, anchor: str) -> str:
    if line in text:
        return text
    if anchor not in text:
        raise PatchError(f"anchor not found while inserting line: {anchor!r}")
    return text.replace(anchor, f"{line}\n{anchor}", 1)


def append_line(text: str, line: str) -> str:
    stripped = text.rstrip("\n")
    if line in text:
        return text
    return f"{stripped}\n{line}\n"


def ensure_block_before(text: str, block: str, anchor: str) -> str:
    if block in text:
        return text
    if anchor not in text:
        raise PatchError(f"anchor not found while inserting block: {anchor!r}")
    return text.replace(anchor, f"{block}\n{anchor}", 1)


def patch_file(path: Path, transform, *, dry_run: bool) -> str:
    original = read_text(path)
    updated = transform(original)
    if updated == original:
        return f"unchanged {display_path(path)}"
    write_text(path, updated, dry_run)
    return f"patch {display_path(path)}"


def action_header_template(name: str, field_type: str, action_type: str, observables: list[tuple[str, str]]) -> str:
    observable_fields = "\n".join(f"        {field_type} {field_name};" for field_name, field_type in observables)
    observable_reset = "\n".join(
        f"            {field_name} = {default_for_type(field_type)};" for field_name, field_type in observables
    )
    observable_plus = "\n".join(f"            {field_name} += rhs.{field_name};" for field_name, _ in observables)
    observable_div = "\n".join(f"            {field_name} /= rhs;" for field_name, _ in observables)
    measure_reset = "\n".join(
        f"    Result.{field_name} = {default_for_type(field_type)};" for field_name, field_type in observables
    )
    return f"""#pragma once

#include <complex>
#include <format>
#include <sstream>
#include <stdexcept>
#include <string>

#include "reticolo/action/actionBase.hpp"
#include "reticolo/core/types/coord.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::action {{

template <RealValue TImpl>
class {name} : public ActionBase<{field_type}, {action_type}, TImpl> {{
  public:
    using base_type = ActionBase<{field_type}, {action_type}, TImpl>;
    using action_type = typename base_type::action_type;
    using field_type = typename base_type::field_type;
    using size_type = typename base_type::size_type;
    using impl_type = TImpl;

    static constexpr int Dims = 4;
    static constexpr int Stencil = 2;

    struct Params {{
        // IMPLEMENTATION REQUIRED:
        // Replace these placeholder parameters with the real couplings for your action.
        impl_type coupling = static_cast<impl_type>(1.0);
        impl_type mass = static_cast<impl_type>(1.0);
    }} p;

    struct Observables {{
{observable_fields}

        void reset() {{
{observable_reset}
        }}

        auto operator+=(const Observables& rhs) -> Observables& {{
{observable_plus}
            return *this;
        }}

        auto operator/=(const impl_type& rhs) -> Observables& {{
{observable_div}
            return *this;
        }}
    }};

    {name}() = default;

    void setup(const YAML::Node& ActionParams) override;
    void lattice_sync(Lattice<field_type>& field) override;

    auto compute_S(Lattice<field_type>& field) -> action_type override;
    auto compute_S_loc(Lattice<field_type>& field, size_type site) -> action_type override;
    auto compute_dS_loc(Lattice<field_type>& field, const field_type& dphi, size_type site) -> action_type override;

    void compute_Forces(Lattice<field_type>& field, Lattice<field_type>& forces) override;
    void compute_LLRForces(Lattice<field_type>& field, Lattice<field_type>& forces, TImpl Sk, TImpl width,
                           TImpl ak) override;

    static auto Measure(Lattice<field_type>& field) -> Observables;

    auto GetName() -> std::string override;
    auto GetParameters() -> std::string override;
}};

template <RealValue TImpl>
inline void {name}<TImpl>::setup(const YAML::Node& ActionParams) {{
    // IMPLEMENTATION REQUIRED:
    // Parse your action-specific YAML fields here.
    if (ActionParams["coupling"]) {{
        p.coupling = ActionParams["coupling"].as<impl_type>();
    }}
    if (ActionParams["mass"]) {{
        p.mass = ActionParams["mass"].as<impl_type>();
    }}
}}

template <RealValue TImpl>
inline void {name}<TImpl>::lattice_sync(Lattice<field_type>& /*field*/) {{
    // IMPLEMENTATION REQUIRED:
    // Precompute any lattice-dependent caches or consistency checks here.
}}

template <RealValue TImpl>
inline auto {name}<TImpl>::compute_S(Lattice<field_type>& /*field*/) -> action_type {{
    // IMPLEMENTATION REQUIRED:
    // Return the total action for the current lattice configuration.
    return action_type{{0}};
}}

template <RealValue TImpl>
inline auto {name}<TImpl>::compute_S_loc(Lattice<field_type>& /*field*/, size_type /*site*/) -> action_type {{
    // IMPLEMENTATION REQUIRED:
    // Return the contribution local to a lattice site if your update algorithm needs it.
    return action_type{{0}};
}}

template <RealValue TImpl>
inline auto {name}<TImpl>::compute_dS_loc(Lattice<field_type>& /*field*/, const field_type& /*dphi*/, size_type /*site*/)
    -> action_type {{
    // IMPLEMENTATION REQUIRED:
    // Return the local action variation associated with a proposed field update.
    return action_type{{0}};
}}

template <RealValue TImpl>
inline void {name}<TImpl>::compute_Forces(Lattice<field_type>& /*field*/, Lattice<field_type>& /*forces*/) {{
    // IMPLEMENTATION REQUIRED:
    // Fill the force lattice if the action supports the generic HMC implementation.
    throw std::runtime_error("{name}::compute_Forces is not implemented");
}}

template <RealValue TImpl>
inline void {name}<TImpl>::compute_LLRForces(Lattice<field_type>& /*field*/, Lattice<field_type>& /*forces*/,
                                             TImpl /*Sk*/, TImpl /*width*/, TImpl /*ak*/) {{
    // IMPLEMENTATION REQUIRED:
    // Fill the force lattice for LLR/HMC-style updates if you plan to support them.
    throw std::runtime_error("{name}::compute_LLRForces is not implemented");
}}

template <RealValue TImpl>
inline auto {name}<TImpl>::Measure(Lattice<field_type>& /*field*/) -> Observables {{
    // IMPLEMENTATION REQUIRED:
    // Populate the observable values that will be written to HDF5.
    Observables Result{{}};
{measure_reset}
    return Result;
}}

template <RealValue TImpl>
inline auto {name}<TImpl>::GetName() -> std::string {{
    std::string Result = "{name}";
    Result += std::same_as<impl_type, RealF> ? " [float]" : " [double]";
    return Result;
}}

template <RealValue TImpl>
inline auto {name}<TImpl>::GetParameters() -> std::string {{
    // IMPLEMENTATION REQUIRED:
    // Replace this with a concise summary of the action parameters.
    return std::format("[ coupling : {{:.3f}}, mass : {{:.3f}} ]", p.coupling, p.mass);
}}

}}  // namespace reticolo::action
"""


def default_for_type(type_name: str) -> str:
    compact = type_name.replace(" ", "")
    if compact in {"float", "double", "impl_type", "TImpl", "RealF", "RealD", "int", "long", "size_t"}:
        return "0"
    if compact.startswith("std::complex<"):
        return "{}"
    return "{}"


def descriptor_block(
    name: str,
    module_name: str,
    storage_schema: str,
    algorithms: list[str],
    custom_algorithms: list[str],
) -> str:
    supports_metropolis = "Metropolis" in algorithms
    supports_hmc = "HMC" in algorithms
    supports_llr = "LLRMetropolis" in algorithms
    alias_comment = (
        "    static constexpr auto             aliases = std::array<std::string_view, 0>{};\n"
        "    // Add aliases above if you want backward-compatible alternate names.\n"
    )
    custom_comment = ""
    if custom_algorithms:
        custom_comment = "    // Custom updater stubs were generated for: " + ", ".join(custom_algorithms) + "\n"
    return f"""struct {descriptor_name(name)} {{
    static constexpr std::string_view module_name = "{module_name}";
    static constexpr std::string_view default_name = "{name}";
    static constexpr std::string_view float_name = "{name}_F";
    static constexpr std::string_view double_name = "{name}_D";
{alias_comment}{custom_comment}    static constexpr bool             supports_metropolis = {bool_literal(supports_metropolis)};
    static constexpr bool             supports_hmc = {bool_literal(supports_hmc)};
    static constexpr bool             supports_llr = {bool_literal(supports_llr)};
    static constexpr bool             has_float_precision = true;
    static constexpr bool             has_double_precision = true;
    static constexpr auto             canonical_precision = ActionPrecisionBinding::double_precision;
    static constexpr auto             alias_precision = canonical_precision;
    static constexpr std::string_view storage_schema = "{storage_schema}";
    static constexpr auto algorithms = std::array<std::string_view, {len(algorithms)}>{{{format_algorithm_array(algorithms)}}};
}};
"""


def format_algorithm_array(algorithms: list[str]) -> str:
    return ", ".join(f'"{algorithm}"' for algorithm in algorithms)


def action_descriptor_specializations(name: str) -> str:
    return f"""template <>
struct ActionDescriptorFor<action::{name}<RealF>> {{
    using type = {descriptor_name(name)};
}};

template <>
struct ActionDescriptorFor<action::{name}<RealD>> {{
    using type = {descriptor_name(name)};
}};
"""


def registration_header_template(name: str) -> str:
    return f"""#pragma once

#include "reticolo/action/{name}.hpp"
#include "reticolo/action/registration/ActionModuleRegistrationSupport.hpp"

namespace reticolo::registration {{

inline void {registration_function_name(name)}() {{
    register_monte_carlo_action_family<{descriptor_name(name)}, action::{name}>();
}}

}}  // namespace reticolo::registration
"""


def storage_header_template(name: str, observables: list[tuple[str, str]]) -> str:
    inserts_float = "\n".join(
        hdf5_insert_line(name, field_name, field_type, "RealF") for field_name, field_type in observables
    )
    inserts_double = "\n".join(
        hdf5_insert_line(name, field_name, field_type, "RealD") for field_name, field_type in observables
    )
    return f"""#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include "reticolo/action/{name}.hpp"
#include "reticolo/core/storage/Hdf5TypeRegistry.hpp"

namespace reticolo {{

template <>
inline auto make_H5_Type<action::{name}<RealF>::Observables>() -> hid_t {{
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::{name}<RealF>::Observables));
{inserts_float}
    return DataTypeHid;
}}

template <>
inline auto make_H5_Type<action::{name}<RealD>::Observables>() -> hid_t {{
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(action::{name}<RealD>::Observables));
{inserts_double}
    return DataTypeHid;
}}

}}  // namespace reticolo
"""


def hdf5_insert_line(name: str, field_name: str, field_type: str, precision: str) -> str:
    h5_type = hdf5_primitive_for(field_type, precision)
    return (
        f'    H5Tinsert(DataTypeHid, "{field_name}", '
        f"HOFFSET(action::{name}<{precision}>::Observables, {field_name}), {h5_type});"
    )


def hdf5_primitive_for(field_type: str, precision: str) -> str:
    compact = field_type.replace(" ", "")
    if compact in {"impl_type", "TImpl"}:
        return "H5T_NATIVE_FLOAT" if precision == "RealF" else "H5T_NATIVE_DOUBLE"
    if compact in {"float", "RealF"}:
        return "H5T_NATIVE_FLOAT"
    if compact in {"double", "RealD"}:
        return "H5T_NATIVE_DOUBLE"
    if compact in {"int"}:
        return "H5T_NATIVE_INT"
    if compact in {"unsigned", "unsignedint"}:
        return "H5T_NATIVE_UINT"
    raise PatchError(
        f"generated HDF5 mapping only supports primitive observable fields. Unsupported field type: {field_type}"
    )


def adapter_header_template(name: str, custom_algorithms: list[str]) -> str:
    blocks: list[str] = []
    if "Metropolis" in custom_algorithms:
        blocks.append(custom_metropolis_block(name, "RealF") + "\n\n" + custom_metropolis_block(name, "RealD"))
    if "HMC" in custom_algorithms:
        blocks.append(custom_hmc_block(name, "RealF") + "\n\n" + custom_hmc_block(name, "RealD"))

    includes = [
        "#include <random>",
        "#include <stdexcept>",
        f'#include "reticolo/action/{name}.hpp"',
    ]
    if "Metropolis" in custom_algorithms:
        includes.append('#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"')
    if "HMC" in custom_algorithms:
        includes.append('#include "reticolo/modules/montecarlo/algorithms/HMC.hpp"')

    joined_includes = "\n".join(includes)
    joined_blocks = "\n\n".join(blocks)
    return f"""#pragma once

{joined_includes}

namespace reticolo {{

{joined_blocks}

}}  // namespace reticolo
"""


def custom_metropolis_block(name: str, precision: str) -> str:
    return f"""template <>
inline void MMonteCarlo::Metropolis<action::{name}<{precision}>>::updateField(
    lattice_type& field,
    action::{name}<{precision}>& action,
    monte_carlo_data_type& state,
    std::mt19937_64& rng) {{
    (void)field;
    (void)action;
    (void)state;
    (void)rng;

    // IMPLEMENTATION REQUIRED:
    // Replace this stub with the custom Metropolis update for {name}<{precision}>.
    throw std::runtime_error("Custom Metropolis adapter for {name}<{precision}> is not implemented");
}}"""


def custom_hmc_block(name: str, precision: str) -> str:
    return f"""template <>
inline void MMonteCarlo::HMC<action::{name}<{precision}>>::updateField(
    Lattice<field_type>& field,
    action::{name}<{precision}>& action,
    monte_carlo_data_type& state,
    std::mt19937_64& rng) {{
    (void)field;
    (void)action;
    (void)state;
    (void)rng;

    // IMPLEMENTATION REQUIRED:
    // Replace this stub with the custom HMC update for {name}<{precision}>.
    throw std::runtime_error("Custom HMC adapter for {name}<{precision}> is not implemented");
}}"""


def sample_yaml_template(
    name: str,
    module_name: str,
    storage_schema: str,
    algorithms: list[str],
    custom_algorithms: list[str],
) -> str:
    preferred_algorithm = "Metropolis" if "Metropolis" in algorithms else algorithms[0]
    algorithm_comment = ""
    if custom_algorithms:
        algorithm_comment = (
            "# This action was generated with custom adapter stubs for: " + ", ".join(custom_algorithms) + "\n"
        )
    algorithm_block = sample_algorithm_block(preferred_algorithm)
    return f"""workflows:
  - {module_name}:
      name: {snake_case(name)}_job
      main_seed: 0

      lattice:
        size: [8, 8, 8, 8]

      action:
        name: {name}
        parameters:
          # IMPLEMENTATION REQUIRED:
          # Replace these placeholders with the parameters parsed in {name}::setup.
          coupling: 1.0
          mass: 1.0

      algorithm:
{algorithm_block}

      output_dir: ./{name}_MonteCarlo
      save_config: true
      save_data: true
      console_output: true

      runs:
        - name: test_run
          measures: 100
          measure_step: 10
          therm_steps: 0
          field_init: true
          hot_start: false

# Storage schema: {storage_schema}
{algorithm_comment}"""


def sample_algorithm_block(algorithm: str) -> str:
    if algorithm == "Metropolis":
        return """        name: Metropolis
        parameters:
          prop_width: 0.1"""
    if algorithm == "HMC":
        return """        name: HMC
        parameters:
          steps: 5
          stepsize: 0.01"""
    if algorithm == "LLRMetropolis":
        return """        name: LLRMetropolis
        parameters:
          # IMPLEMENTATION REQUIRED:
          # Fill the LLR parameters expected by your updater implementation.
          prop_width: 0.1
          Sk: 0.0
          ak: 0.0
          width: 1.0"""
    raise PatchError(f"unsupported sample algorithm: {algorithm}")


def registry_metadata_patch(name: str, module_name: str, algorithms: list[str]) -> tuple[str, str]:
    include_line = f'#include "reticolo/action/{name}.hpp"'
    test_block = f"""
    using {name}Action = reticolo::action::{name}<reticolo::RealD>;
    using {name}Handler = reticolo::MMonteCarlo::MonteCarloHandler<{name}Action>;
"""
    algorithm_assertions = "\n".join(
        f'        require_contains({snake_case(name)}_info->algorithms, "{algorithm}", "missing {algorithm} for {name}");'
        for algorithm in algorithms
    )
    runtime_block = f"""
        require_contains(mc_actions, "{name}", "missing {name} action");
        require_contains(mc_actions, "{name}_F", "missing {name}_F action");
        require_contains(mc_actions, "{name}_D", "missing {name}_D action");

        const auto {snake_case(name)}_info = reticolo::runtime::metadata::describe_action("{name}");
        require({snake_case(name)}_info.has_value(), "missing {name} metadata");
        require({snake_case(name)}_info->canonical_name == "{name}", "wrong canonical name for {name}");
        require({snake_case(name)}_info->canonical_precision == reticolo::registration::ActionPrecisionBinding::double_precision,
                "{name} canonical precision should be explicit and double");
{algorithm_assertions}

        auto {snake_case(name)}_module = reticolo::ModuleFactory::MakeModule("{module_name}", "{name}");
        require(static_cast<bool>({snake_case(name)}_module), "failed to create {name} module");
        require(dynamic_cast<{name}Handler*>({snake_case(name)}_module.get()) != nullptr,
                "{name} canonical action should bind to the declared canonical precision module");

        auto {snake_case(name)}_updater =
            reticolo::MMonteCarlo::AlgorithmFactory::MakeUpdater<{name}Action>("{algorithms[0]}");
        require(static_cast<bool>({snake_case(name)}_updater), "failed to create {name} updater");
"""
    return include_line + "\n", test_block + "\n", runtime_block + "\n"


def apply_registry_metadata_patch(contents: str, name: str, module_name: str, algorithms: list[str]) -> str:
    include_line, aliases_block, runtime_block = registry_metadata_patch(name, module_name, algorithms)
    contents = ensure_line(
        contents,
        include_line.rstrip("\n"),
        '#include "reticolo/action/WeakFieldEuclideanGR.hpp"',
    )
    contents = ensure_block_before(
        contents,
        aliases_block.rstrip("\n"),
        "    try {",
    )
    contents = ensure_block_before(
        contents,
        runtime_block.rstrip("\n"),
        '        const auto wfgr_info = reticolo::runtime::metadata::describe_action("WeakFieldEuclideanGR");',
    )
    return contents


def make_changes(args: argparse.Namespace, paths: ProjectPaths) -> list[str]:
    action_name = args.name
    storage_schema = args.storage_schema or snake_case(action_name)
    algorithms = parse_csv(args.algorithms)
    custom_algorithms = parse_csv(args.custom_algorithms)
    observables = parse_observables(args.observable_fields)

    if not algorithms:
        raise SystemExit("at least one algorithm must be selected")

    for algorithm in algorithms:
        if algorithm not in VALID_ALGORITHMS:
            raise SystemExit(f"unsupported algorithm '{algorithm}'")

    for algorithm in custom_algorithms:
        if algorithm not in CUSTOMIZABLE_ALGORITHMS:
            raise SystemExit(f"custom adapter generation is only supported for {', '.join(CUSTOMIZABLE_ALGORITHMS)}")
        if algorithm not in algorithms:
            raise SystemExit(f"custom algorithm '{algorithm}' must also appear in --algorithms")

    changes: list[str] = []

    action_file = paths.action_dir / f"{action_name}.hpp"
    registration_file = paths.registration_dir / f"{action_name}ModuleRegistration.hpp"
    storage_file = paths.storage_dir / f"{action_name}Hdf5.hpp"
    yaml_file = paths.scripts_input_dir / f"{action_name}_MonteCarlo.yaml"
    adapter_file = paths.adapter_dir / f"{action_name}MonteCarlo.hpp"

    changes.append(
        create_file(
            action_file,
            action_header_template(action_name, args.field_type, args.action_type, observables),
            force=args.force,
            dry_run=args.dry_run,
        )
    )
    changes.append(
        create_file(
            registration_file,
            registration_header_template(action_name),
            force=args.force,
            dry_run=args.dry_run,
        )
    )
    changes.append(
        create_file(
            storage_file,
            storage_header_template(action_name, observables),
            force=args.force,
            dry_run=args.dry_run,
        )
    )
    changes.append(
        create_file(
            yaml_file,
            sample_yaml_template(action_name, args.module_name, storage_schema, algorithms, custom_algorithms),
            force=args.force,
            dry_run=args.dry_run,
        )
    )

    if custom_algorithms:
        changes.append(
            create_file(
                adapter_file,
                adapter_header_template(action_name, custom_algorithms),
                force=args.force,
                dry_run=args.dry_run,
            )
        )

    action_include = f'#include "reticolo/action/{action_name}.hpp"'
    descriptor_struct = descriptor_block(
        action_name,
        args.module_name,
        storage_schema,
        algorithms,
        custom_algorithms,
    )
    specialization_block = action_descriptor_specializations(action_name)

    changes.append(
        patch_file(
            paths.action_descriptor,
            lambda text: ensure_block_before(
                ensure_block_before(
                    ensure_line(text, action_include, '#include "reticolo/action/WeakFieldEuclideanGR.hpp"'),
                    descriptor_struct.rstrip("\n"),
                    "template <typename Action>\nstruct ActionDescriptorFor;",
                ),
                specialization_block.rstrip("\n"),
                "template <typename Action>\nusing action_descriptor_t = ActionDescriptorFor<Action>::type;",
            ),
            dry_run=args.dry_run,
        )
    )

    changes.append(
        patch_file(
            paths.builtin_action_families,
            lambda text: append_line(
                text,
                f"RETICOLO_BUILTIN_ACTION_FAMILY({descriptor_name(action_name)}, {registration_function_name(action_name)})",
            ),
            dry_run=args.dry_run,
        )
    )

    changes.append(
        patch_file(
            paths.builtin_module_registration,
            lambda text: ensure_line(
                text,
                f'#include "reticolo/action/registration/{action_name}ModuleRegistration.hpp"',
                '#include "reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp"',
            ),
            dry_run=args.dry_run,
        )
    )

    changes.append(
        patch_file(
            paths.hdf5_type_mappings,
            lambda text: ensure_line(
                text,
                f'#include "reticolo/action/storage/{action_name}Hdf5.hpp"',
                '#include "reticolo/action/storage/WeakFieldEuclideanGRHdf5.hpp"',
            ),
            dry_run=args.dry_run,
        )
    )

    if custom_algorithms:
        changes.append(
            patch_file(
                paths.builtin_mc_adapters,
                lambda text: ensure_line(
                    text,
                    f'#include "reticolo/action/adapters/{action_name}MonteCarlo.hpp"',
                    '#include "reticolo/action/adapters/WeakFieldEuclideanGRMonteCarlo.hpp"',
                ),
                dry_run=args.dry_run,
            )
        )

    changes.append(
        patch_file(
            paths.registry_metadata_test,
            lambda text: apply_registry_metadata_patch(text, action_name, args.module_name, algorithms),
            dry_run=args.dry_run,
        )
    )

    return changes


def main() -> int:
    args = parse_args()
    interactive_mode = args.interactive or args.name is None
    if interactive_mode:
        args = run_interactive_wizard(args)

    if args.name is None:
        raise SystemExit("action name is required")

    ensure_action_name(args.name)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    paths = project_paths(root)

    required_files = (
        paths.action_descriptor,
        paths.builtin_action_families,
        paths.builtin_module_registration,
        paths.builtin_mc_adapters,
        paths.hdf5_type_mappings,
        paths.registry_metadata_test,
    )
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("project layout check failed, missing files:\n" + "\n".join(missing))

    try:
        changes = make_changes(args, paths)
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mode = "Dry run" if args.dry_run else "Generated"
    print(f"{mode} action scaffold for {args.name}")
    for change in changes:
        print(f" - {change}")

    if interactive_mode and args.dry_run:
        print()
        print("Preview complete.")
        if prompt_yes_no("Write these files now using the same answers?", default=False):
            args.dry_run = False
            try:
                changes = make_changes(args, paths)
            except PatchError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"Generated action scaffold for {args.name}")
            for change in changes:
                print(f" - {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
