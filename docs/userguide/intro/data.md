# Data Movement

:::{admonition} Keeping Data Transparent
:class: tip
"Show me your code and conceal your data structures, and I shall continue to be
mystified. Show me your data structures, and I won't usually need your code; it'll be
obvious." - *Fred Brooks*
:::

Earth-2 Inference Studio aims to keep data simple and interpretable between components.
Given that this package interacts with geo-physial data, the common data-structure
inside workflows is the pairing of:

1. A PyTorch tensor (`torch.Tensor`) on the inference device holding the array data of
interest.
2. An OrderedDict of numpy arrays (`CoordSystem`) that
represents the geophyiscal coordinate system the tensor represents.

For example, perturbation methods operate by using a data tensor and coordinate system
to generate a noise tensor:

```python
class PerturbationMethod(Protocol):
    """Perturbation interface."""

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : CoordSystem
            Ordered dict representing coordinate system that discribes the tensor

        Returns
        -------
        torch.Tensor
            Perturbation noise tensor
        """
        pass
```

In later sections, users will find that most components have APIs that either generate
or interact with these two data structures.
The combonation of both the data tensor and respective coordinate system provides
complete information one needs to interpret any stage of a workflow.

## Coordinate Systems

As previously discussed, coordinate dictionaries are a critical part of Earth-2
Inference Studio's data movement.
We wanted the coordinate object to be a fairly primative data object that allows users
to interact with the data outside the project and keep things transparent in workflows.
Inside the package these are typed as `CoordSystem` which is defined as the following:

```python
CoordSystem = NewType("CoordSystem", OrderedDict[str, np.ndarray])
```

The dictionary is ordered since the keys correspond the the dimensions of the associated
data tensor.
Let's consider a simple example of a 2D lat-lon grid:

```python
x = torch.randn(181, 360)

coords = OrderedDict({
    "lat": np.linspace(-90, 90, 181),
    "lon": np.linspace(0, 360, 360, endpoint=False)
})
```

Much of Earth-2 Inference Studio typically operates on a lat-lon grid but it's not
required to.

### Standard Coordinate Names

Earth-2 Inference Studio has a dimension naming standard for its built in feature set.
We encourage users to follow similar naming schemes for compatability between Earth-2
Inference Studio when possible and the packages we interface with.

```{list-table}
    :widths: 15 40 25
    :header-rows: 1

   * - Key
     - Description
     - Type
   * - `batch`
     - Dimension representing the batch dimension of the data. Used to denote a "free"
     dimension, consult batching docs for more details.
     - `np.empty(1)`
   * - `time`
     - Time dimension, represented via numpy arrays of datetime objects.
     - `np.ndarray[np.datetime64[ns]]` (`TimeArray`)
   * - `lead_time`
     - Lead time is used to denote a dimension that indexes over forecast steps.
     - `np.ndarray[np.timedelta64[ns]]` (`LeadTimeArray`)
   * - `variable`
     - Dimension representing physical variable (atmospheric, surface, etc). Earth-2
     Inference Studio has its own naming convention. See lexicon docs more more details.
     - `np.ndarray[str]` (`VariableArray`)
   * - `lat`
     - Lattitude coordinate array
     - `np.ndarray[float]`
   * - `lat`
     - Longitude coordinate array
     - `np.ndarray[float]`
```

### Coordinate Utilities

The downside of using a dictionary to store coordinates is that manipulating the data
tensor and then updating the coordinate array is not a manual process.
To help make this process less tedious, Earth-2 Inference Studio has several utility
functions that make interacting with coordinates easier.
The bulk of these can be found in the [Earth-2 Studio Utilities](earth2studio.utils_api).

:::{warning}
🚧 Under construction, todo: add some example here! 🚧
:::

## Inference on the GPU

It is beneficial to leverage the GPU for as many processes as possible.
Earth-2 Inference Studio aims to get data from the data source and immediately convert
it into the tensor, coord data struction on the device.
From there, the data is kept on the GPU until the very last moment when writes are
needed to in-memory or to file.

```{figure} https://gitlab-master.nvidia.com/ngeneva/earth-2-studio-assets/-/raw/main/e2studio-data.png
:alt: earth2studio-data
:width: 600px
:align: center
```

In the figure above, that the data is first pulled from the data source as an Xarray
data array which is then then converted to a tensor.
The data remain on the device, denoted by the GPU boundary, until it needs to be written
by the IO component.

:::{admonition} Data Sources and Xarray
:class: tip
This may raise the question: Why do datasources not output directly to tensor and
coordinate dictionaries?
This is an opinionated decision due to the fact that these data sources need to store
data on the CPU regardless and can be extremely useful outside of the context of this package.
Thus they return Xarray data arrays which are is nothing more than a fancy data array
with a coordinate system attached to it!
:::