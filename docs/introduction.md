# Welcome to LightningLite

LightningLite is a project that realizes [Pytorch Lightning](https://lightning.ai/)-like framework in minimal, readable and customizable codes. Due to the black-box properties of Pytorch Lightning, it turns out to be servel inevitable problems for researchers:

1. When encounter errors, it's sometimes catastrophic because of the coupling properties of Pytorch Lighning.
2. You can't add or delete codes to build up your own codebase.
3. It may be no suitable for beginner who need to understand all the codes.

LightningLite addresses these issues by seperating it into 3 fully decouping main conponents, which each codes are less than 150 lines.

For full documentation visit [github](https://github.com/Yancy456/LightningLite).

## Main components

LightningLite implements 3 necessary components for any deep learning experiment.

1. Logger: Logging system is the key to any scientific experiment. It allows researchs to persist experimental data for later research. This will be very helpful when you want to draw some charts for your experiments.
2. Printer: Beside Logger, researchs wants to know real-time results of current experiment. Printer formats all results and puts them on console.
3. Timer: Time is also a key to scientific experiment. It give you full control about how long the experiment will take.

## Wrapper classes

In order to orgnize above components and some automatic features into an experiment, LightningLite implements 3 following wrapper classes.

1. LiteModule: A wrapped class for augmenting the abilities of original torch module by adding more details of how to do before/on/after training/validation/test stages.
2. DataModule: A wrapped class which tells the details of how to load dataset for training/validation/test.
3. Trainer: A general training processor, including automatic features of loading data and models to accelerators.

## installation

Install from souce.
```
cd <your_project>
git clone https://github.com/Yancy456/LightningLite.git
```
