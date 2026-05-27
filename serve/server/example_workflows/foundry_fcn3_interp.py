# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.io import IOBackend
from earth2studio.models.dx import DerivedSurfacePressure
from earth2studio.models.px import FCN3, DiagnosticWrapper, InterpModAFNO
from earth2studio.serve.server import (
    WorkflowProgress,
    WorkflowRegistry,
)
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

from .foundry_fcn3 import FoundryFCN3Workflow


@WorkflowRegistry.instance().register
class FoundryFCN3InterpWorkflow(FoundryFCN3Workflow):
    """FCN3 ensemble inference workflow for Foundry using ECMWF IFS initial conditions."""

    name = "foundry_fcn3_interp_workflow"
    description = "FCN3 ensemble workflow with hourly interpolation for Foundry"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_fcn3(self) -> InterpModAFNO:
        logger.info("Loading FCN3 with InterpModAFNO")
        fcn3_package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(fcn3_package)

        # FCN3 does not output surface pressure, but we can derive it from other variables
        orography_fn = fcn3_package.resolve("orography.nc")
        with xr.open_dataset(orography_fn) as ds:
            z_surface = torch.as_tensor(ds["Z"][0].values)
        z_surf_coords = CoordSystem({d: fcn3.input_coords()[d] for d in ["lat", "lon"]})
        sp_model = DerivedSurfacePressure(
            p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            surface_geopotential=z_surface,
            surface_geopotential_coords=z_surf_coords,
        )

        fcn3_with_sp = DiagnosticWrapper(px_model=fcn3, dx_model=sp_model)

        # Add the time interpolation model to go from 6-hourly to hourly
        interp_package = InterpModAFNO.load_default_package()
        fcn3_with_sp_interp = InterpModAFNO.load_model(
            interp_package, px_model=fcn3_with_sp
        )

        fcn3_with_sp_interp.to(self.device)
        fcn3_with_sp_interp.eval()
        return fcn3_with_sp_interp

    def __call__(
        self,
        io: IOBackend,
        start_time: datetime = datetime(2025, 1, 1),
        n_steps: int = 12,
        n_samples: int = 4,
        seeds: Sequence[int] | None = None,
        variables: Sequence[str] | None = ("t2m", "u10m", "v10m"),
        output_format: Literal["zarr", "netcdf4"] = "netcdf4",
        container_url: str | None = None,
        geo_catalog_url: str | None = None,
        collection_id: str | None = None,
    ) -> None:
        self.validate_start_time(start_time)
        lead_times = np.array([np.timedelta64(i, "h") for i in range(n_steps + 1)])
        seeds = self.validate_samples(n_samples, seeds)
        variables = self.validate_variables(variables)

        x_ori, coords_ori = self.get_fcn3_input(start_time)

        output_coords = CoordSystem(
            {
                "ensemble": np.arange(len(seeds)),
                # Combine 'time' and 'lead_time' into single dimension
                "time": to_time_array([start_time]) + lead_times,
                "variable": variables,
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(-180, 180, 1440, endpoint=False),
            }
        )
        self.setup_io(io, output_coords, seeds)

        logger.info("Starting inference")
        total_samples = len(seeds)
        n_steps += 1  # add 1 for step 0 (initial conditions)
        for sample, seed in enumerate(seeds):

            self.fcn3.px_model.px_model.set_rng(seed=seed)
            iterator = self.fcn3.create_iterator(x_ori.clone(), coords_ori.copy())
            for step, (x, coords) in enumerate(iterator):
                # Update progress for step within sample
                msg = (
                    f"Processing sample {sample + 1}/{total_samples} "
                    f"(seed={seed}), step {step + 1}/{len(lead_times)}"
                )
                progress = WorkflowProgress(
                    progress=msg,
                    current_step=step + 1,
                    total_steps=n_steps,
                )
                self.update_progress(progress)
                logger.info(msg)

                # Select variables
                x_out, coords_out = map_coords(
                    x, coords, CoordSystem({"variable": output_coords["variable"]})
                )
                # Roll longitudes (for raster visualization)
                x_out = torch.roll(x_out, 720, dims=-1)
                coords_out["lon"] = np.linspace(-180, 180, 1440, endpoint=False)
                # Add ensemble dimension
                x_out = x_out.unsqueeze(0)
                coords_out["ensemble"] = np.array([sample])
                coords_out.move_to_end("ensemble", last=False)
                # Combine time and lead_time
                lead_time_dim = list(coords_out).index("lead_time")
                x_out = x_out.squeeze(lead_time_dim)
                coords_out["time"] = coords_out["time"] + coords_out["lead_time"]
                del coords_out["lead_time"]
                # Write to disk
                io.write(*split_coords(x_out, coords_out))

                if step == (n_steps - 1):
                    break
