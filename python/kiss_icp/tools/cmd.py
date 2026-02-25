# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import typer

from kiss_icp.datasets import (
    available_dataloaders,
    jumpable_dataloaders,
    sequence_dataloaders,
    supported_file_extensions,
)


def guess_dataloader(data: Path, default_dataloader: str):
    """Guess which dataloader to use in case the user didn't specify with --dataloader flag."""
    if data.is_file():
        if data.name == "metadata.yaml":
            return "rosbag", data.parent
        if data.name.split(".")[-1] in "bag":
            return "rosbag", data
        if data.name.split(".")[-1] == "pcap":
            return "ouster", data
        if data.name.split(".")[-1] == "mcap":
            return "mcap", data
    elif data.is_dir():
        if (data / "metadata.yaml").exists():
            return "rosbag", data
        if len(glob.glob(os.path.join(data, "*.bag"))) > 0:
            return "rosbag", data
    return default_dataloader, data


def version_callback(value: bool):
    if value:
        try:
            from kiss_icp.pybind import kiss_icp_pybind
        except ImportError as e:
            print(80 * "*")
            print(f"[ERRROR] Python bindings not properly built! Please open a issue on github")
            print(f"[ERRROR] '{e}'")
            print(80 * "*")
            raise typer.Exit(1)

        import kiss_icp
        print(f"KISS-ICP Version: {kiss_icp.__version__}")
        raise typer.Exit(0)


def name_callback(value: str):
    if not value:
        return value
    dl = available_dataloaders()
    if value not in dl:
        raise typer.BadParameter(f"Supported dataloaders are:\n{', '.join(dl)}")
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

_available_dl_help = available_dataloaders()
_available_dl_help.remove("generic")
_available_dl_help.remove("mcap")
_available_dl_help.remove("ouster")
_available_dl_help.remove("rosbag")

docstring = f"""
:kiss: KISS-ICP, a simple yet effective LiDAR-Odometry estimation pipeline :kiss:\n
\b
[bold green]Examples: [/bold green]
# Process all pointclouds in the given <data-dir> \[{", ".join(supported_file_extensions())}]
$ kiss_icp_pipeline --visualize <data-dir>:open_file_folder:

# Custom adaptive params
$ kiss_icp_pipeline --visualize <data-dir> --base 0.08 0.002 0.15
"""


@app.command(help=docstring)
def kiss_icp_pipeline(
    data: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    dataloader: str = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="[Optional] Use a specific dataloader from those supported by KISS-ICP",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="[Optional] Open an online visualization of the KISS-ICP pipeline",
        rich_help_panel="Additional Options",
    ),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        "-t",
        show_default=False,
        help="[Optional] Only valid when processing rosbag files",
        rich_help_panel="Additional Options",
    ),
    n_scans: int = typer.Option(
        -1,
        "--n-scans",
        "-n",
        show_default=False,
        help="[Optional] Specify the number of scans to process, default is the entire dataset",
        rich_help_panel="Additional Options",
    ),
    jump: int = typer.Option(
        0,
        "--jump",
        "-j",
        show_default=False,
        help="[Optional] Specify if you want to start to process scans from a given starting point",
        rich_help_panel="Additional Options",
    ),
    meta: Optional[Path] = typer.Option(
        None,
        "--meta",
        "-m",
        exists=True,
        show_default=False,
        help="[Optional] For Ouster pcap dataloader, specify metadata json file path explicitly",
        rich_help_panel="Additional Options",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show the current version of KISS-ICP",
        callback=version_callback,
        is_eager=True,
    ),
    adaptive_params: Optional[Tuple[float, float, float]] = typer.Option(
        None,
        "--base",
        help="[Optional] Override Adaptive Planarity: <base> <min_thr> <max_thr>",
        rich_help_panel="Hybrid Options",
        show_default=False,
    ),
    # [THÊM MỚI] Khai báo cờ --mode
    reg_mode: Optional[int] = typer.Option(
        None,
        "--mode",
        help="[Optional] ICP Mode. 0: Hybrid, 1: Point-to-Point, 2: Point-to-Plane",
        rich_help_panel="Hybrid Options",
        show_default=False,
    ),
):
    # Attempt to guess some common file extensions
    if not dataloader:
        dataloader, data = guess_dataloader(data, default_dataloader="generic")

    # Validate some options
    if dataloader in sequence_dataloaders() and sequence is None:
        print('You must specify a sequence "--sequence"')
        raise typer.Exit(code=1)

    if jump != 0 and dataloader not in jumpable_dataloaders():
        print(f"[WARNING] '{dataloader}' does not support '--jump', starting from first frame")
        jump = 0

    from kiss_icp.datasets import dataset_factory
    from kiss_icp.pipeline import OdometryPipeline
    from kiss_icp.config import load_config

    # 1. Load cấu hình (lấy mặc định từ config.py hoặc file YAML)
    pipeline_config = load_config(config)

    # 2. Ghi đè tham số nếu người dùng nhập --base từ terminal
    if adaptive_params is not None:
        pipeline_config.adaptive_threshold.adaptive_base = adaptive_params[0]
        pipeline_config.adaptive_threshold.min_planarity_thr = adaptive_params[1]
        pipeline_config.adaptive_threshold.max_planarity_thr = adaptive_params[2]
        print(f"[INFO] Using CLI Adaptive Params: base={adaptive_params[0]}, min={adaptive_params[1]}, max={adaptive_params[2]}")

    # [THÊM MỚI] Ghi đè tham số nếu người dùng nhập --mode từ terminal
    if reg_mode is not None:
        pipeline_config.registration.reg_mode = reg_mode

    OdometryPipeline(
        dataset=dataset_factory(
            dataloader=dataloader,
            data_dir=data,
            sequence=sequence,
            topic=topic,
            meta=meta,
        ),
        config=pipeline_config,
        visualize=visualize,
        n_scans=n_scans,
        jump=jump,
    ).run().print()


def run():
    app()
