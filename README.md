# gsp-ebook-exercises

Este repositório reúne dois exercícios, com suas soluções, do e-book sobre **Processamento de Sinais sobre Grafos** (GSP, _graph signal processing_) das [Notas em Matemática Aplicada, Volume 92/2021](https://proceedings.science/notas-sbmac), publicado pela Sociedade Brasileira de Matemática Aplicada e Computacional – SBMAC.

Os autores deste e-book são:

- Juliano Bandeira Lima ([julianobandeira@ieee.org](mailto:julianobandeira@ieee.org)) *
- Guilherme Boaviagem Ribeiro ([guilherme.boaviagem@gmail.com](mailto:guilherme.boaviagem@gmail.com)) *
- Wallace Alves Martins ([wallace.martins@smt.ufrj.br](mailto:wallace.martins@smt.ufrj.br)) **
- Vitor Rosa Meireles Elias ([vtrmeireles@poli.ufrj.br](mailto:vtrmeireles@poli.ufrj.br)) **
- Gabriela Lewenfus ([gabriela.lewenfus@gmail.com](mailto:gabriela.lewenfus@gmail.com)) **

\* Departamento de Eletrônica e Sistemas, Centro de Tecnologia e Geociências, Universidade Federal de Pernambuco.

** Departamento de Engenharia Eletrônica e de Computação, Escola Politécnica, Universidade Federal do Rio de Janeiro.

## Dependências

Para executar os códigos deste repositório, o ambiente Python precisa conter os pacotes listados em [`requirements.txt`](requirements.txt). Recomenda-se criar um ambiente Python dedicado para isso. Por exemplo, utilizando [Conda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) no seu terminal,
```sh
conda create --name gsp_ebook python=3.7
```
Depois de criado o ambiente, acesse-o com `conda activate gsp_ebook`.

Clone este repositório,
```sh
git clone https://github.com/gboaviagem/gsp-ebook-exercises.git
```

Instale todas as dependências:
```sh
python -m pip install -r gsp-ebook-exercises/requirements.txt
```
