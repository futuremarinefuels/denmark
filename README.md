# Future Marine Fuels
## A Danish Case Study on Climate Compatible Energy Pathways

The research project's main objective is to minimize total system costs for supplying all Danish cargo shipping transport demand while meeting emission reduction target and budget. It investigates possible future energy pathways until 2050 towards carbon neutral Danish shipping in line with the Paris agreement. We consider the following technical and fuel alternatives to the most commonly used internal combustion engines (IC) that burn heavy fuel (HFO) or marine diesel oil (MDO): Bio-diesel (BDO), liquefied bio-methane (LBG), liquefied natural gas (LNG), methane (CH4), methanol (CH3OH), hydrogen (H2), ammonia (NH3) and batteries/wind (BATW).

The problem is translated into a stock model that comprises relevant characteristics of today’s Danish cargo fleet, its emissions, transport work and fuel consumption. It has different alternative technology investment options as well as emission transport constraints.


### Installation

1. Install [Anaconda](https://anaconda.org/) as package and environment manager
2. Create an environment with Anaconda from the .yml file.

## Getting started
Open cmd and run jupyter notebook to investigate the notebooks and run spyder to open the model.

### Pre-processing
The preprocessing notebook creates the input data for the model.

1. Open cmd and run jupyter notebook.
2. Open the pre-processing script.
3. Choose a cost variation rate (default is zero) and run the notebook.
4. Step 3 can be repeated with different rates.

### Model excecution
The model loads the data from the pre-processing depending on a selected rate and scenario.

1. Open cmd and run spyder.
2. Open model.py.
3. Select rate and scenario.
4. Excecute full script.
5. Step 3 and 4 can be repeated with different rates and scenarios.

### Post-processing
The post-processing creates a dataframe with the results for a selected model run in csv format.

1. Return to jupyter.
2. Open the post-processing script.
3. Select rate and scenario and run the notebook.
4. Step 3 can be repeated several times with all rate and scenario combinations calculated in __Model excecution__.


## Built With

* [Python](https://www.python.org/) - Coding language
* [PYOMO](http://www.pyomo.org/) - Modelling environment
* [GLPK](https://www.gnu.org/software/glpk/) - Linear Solver
* [Spyder](https://www.spyder-ide.org/) - Integrated development environement
* [Jupyter Notebook](http://jupyter.org/) - Data processing
* [Anaconda](https://anaconda.org/) - Package and environment management

## License

The repository is published under the [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

