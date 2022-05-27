<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/puchee99/CaixaBank_DS">
    <img src="images/caixabank.jpeg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">CaixaBank-DS-2022</h3>

  <p align="center">
    Online Data Science hackathon  (CaixaBank 2022)
    <br />
    <a href="https://github.com/puchee99/CaixaBank_DS"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/puchee99/CaixaBank_DS">View Demo</a>
    ·
    <a href="https://github.com/puchee99/CaixaBank_DS/issues">Report Bug</a>
    ·
    <a href="https://github.com/puchee99/CaixaBank_DS/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
      <li><a href="#built-with">Built With</a></li>
      <li><a href="#eda">EDA</a></li>
      <li><a href="#model">Model</a></li>
      <li><a href="#results">Results</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

The challenge will consist of modeling a predictive algorithm that allows knowing the future dynamics of the market from the historical data of the value of the IBEX35 and some Tweets.



<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [Pytorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Seaborn](https://seaborn.pydata.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

## EDA



<p align="right">(<a href="#top">back to top</a>)</p>

## Results




<!-- GETTING STARTED -->
## Getting Started

### Installation


First, clone the repository:
   ```sh
   git clone https://github.com/puchee99/CaixaBank_DS.git
   ```
Access to the project folder with:
  ```sh
  cd CaixaBank_DS
  ```

We will create a virtual environment with `python3`
* Create environment with python 3 
    ```sh
    python3 -m venv venv
    ```
    
* Enable the virtual environment
    ```sh
    source venv/bin/activate
    ```

* Install the python dependencies on the virtual environment
    ```sh
    pip install -r requirements.txt
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

The main.ipynb document contains all the steps to get to the solution.
This uses functions from the personal utils library.

The `train.py` and `test.py` documents can be executed with bash using different arguments.

* To get the information of the arguments use:
    ```sh
    python name_document.py -h
    ```
    Example:
    ```sh
    python train.py -h
    ```
* To train the models use:
    ```sh
    python train.py
    ```
* To test the models use:
    ```sh
    python test.py
    ```

<!-- CONTACT -->
## Contact

Arnau Puche  - [@arnau_puche_vila](https://www.linkedin.com/in/arnau-puche-vila-ds/) - arnaupuchevila@gmail.com

Project Link: [https://github.com/puchee99/CaixaBank_DS](https://github.com/puchee99/CaixaBank_DS)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/puchee99/CaixaBank_DS.svg?style=for-the-badge
[contributors-url]: https://github.com/puchee99/CaixaBank_DS/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/puchee99/CaixaBank_DS.svg?style=for-the-badge
[forks-url]: https://github.com/puchee99/CaixaBank_DS/network/members
[stars-shield]: https://img.shields.io/github/stars/puchee99/CaixaBank_DS.svg?style=for-the-badge
[stars-url]: https://github.com/puchee99/CaixaBank_DS/stargazers
[issues-shield]: https://img.shields.io/github/issues/puchee99/CaixaBank_DS.svg?style=for-the-badge
[issues-url]: https://github.com/puchee99/CaixaBank_DS/issues
[license-shield]: https://img.shields.io/github/license/puchee99/CaixaBank_DS.svg?style=for-the-badge
[license-url]: https://github.com/puchee99/CaixaBank_DS/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/arnau-puche-vila-ds/
