# Documentação Técnica: Sistema de Detecção de Envenenamento

## Sumário

### 1. Arquitetura da Rede Neural
- [1.1 Classe Net (modelNet.py)](#11-classe-net-modelnetpy)
  - [1.1.1 Inicialização da Rede](#111-inicialização-da-rede)
  - [1.1.2 Método forward()](#112-método-forward)
  - [1.1.3 Método predict() para LIME](#113-método-predict-para-lime)
- [1.2 Fluxo de Dados Completo](#12-fluxo-de-dados-completo)

### 2. Conceitos Fundamentais do PyTorch
- [2.1 Camadas Convolucionais (Conv2d)](#21-camadas-convolucionais-conv2d)
- [2.2 Regularização com Dropout](#22-regularização-com-dropout)
- [2.3 Camadas Lineares (Linear)](#23-camadas-lineares-linear)
- [2.4 Funções de Ativação](#24-funções-de-ativação)
  - [2.4.1 ReLU](#241-relu)
  - [2.4.2 Softmax e LogSoftmax](#242-softmax-e-logsoftmax)
- [2.5 Max Pooling](#25-max-pooling)
- [2.6 Normalização de Dados](#26-normalização-de-dados)

---

## 1. Arquitetura da Rede Neural

### 1.1 Classe Net (modelNet.py)

A classe `Net` implementa uma Rede Neural Convolucional (CNN) para classificação de dígitos MNIST. Esta arquitetura segue o padrão clássico de extração hierárquica de características seguida por classificação.

**Estrutura Geral:**
```
Entrada (28x28) → Conv1 → ReLU → MaxPool → Conv2 → Dropout → ReLU → MaxPool → Flatten → FC1 → ReLU → Dropout → FC2 → LogSoftmax
```

#### 1.1.1 Inicialização da Rede

```python
def __init__(self):
    super(Net, self).__init__()
    
    # Camadas convolucionais
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d(p=0.5)
    
    # Camadas totalmente conectadas
    self.fc1 = nn.Linear(in_features=320, out_features=50)
    self.fc2 = nn.Linear(in_features=50, out_features=10)
```

**Componentes:**

1. **conv1**: Primeira [camada convolucional](#21-camadas-convolucionais-conv2d)
   - Detecta características básicas (bordas, linhas)
   - Transforma: (1,28,28) → (10,24,24)

2. **conv2**: Segunda [camada convolucional](#21-camadas-convolucionais-conv2d)
   - Detecta padrões complexos (formas, curvas)
   - Transforma: (10,12,12) → (20,8,8)

3. **conv2_drop**: [Dropout 2D](#22-regularização-com-dropout) para regularização
   - Previne overfitting zerando mapas aleatoriamente

4. **fc1**: Primeira [camada linear](#23-camadas-lineares-linear)
   - Integra características: 320 → 50 neurônios
   - **Cálculo do 320**: 20 canais × 4×4 pixels = 320

5. **fc2**: [Camada linear](#23-camadas-lineares-linear) de classificação
   - Produz saídas finais: 50 → 10 classes (dígitos 0-9)

#### 1.1.2 Método forward()

Define como os dados fluem através da rede durante a inferência:

```python
def forward(self, x):
    # Primeiro bloco convolucional
    x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
    
    # Segundo bloco convolucional com regularização
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
    
    # Achatar para camadas densas
    x = x.view(-1, 320)
    
    # Camadas totalmente conectadas
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    
    return F.log_softmax(x, dim=1)
```

**Transformações por etapa:**
1. **Entrada**: (batch_size, 1, 28, 28)
2. **Após conv1+pool**: (batch_size, 10, 12, 12)
3. **Após conv2+pool**: (batch_size, 20, 4, 4)
4. **Após flatten**: (batch_size, 320)
5. **Após fc1**: (batch_size, 50)
6. **Saída final**: (batch_size, 10)

#### 1.1.3 Método predict() para LIME

Interface especial para integração com LIME (Local Interpretable Model-agnostic Explanations):

```python
def predict(self, images):
    # Pré-processamento automático
    transform = transforms.Compose([...])
    
    # Conversão e processamento em lote
    batch = torch.stack([transform(img) for img in images])
    
    # Inferência sem gradientes
    with torch.no_grad():
        log_probs = self.forward(batch)
        probs = F.softmax(log_probs, dim=1)
    
    return probs.cpu().numpy()
```

**Características especiais:**
- Aceita lista de imagens numpy
- Aplica [normalização MNIST](#26-normalização-de-dados) automaticamente
- Retorna probabilidades (não log-probabilidades)
- Otimizado para análise de explicabilidade

### 1.2 Fluxo de Dados Completo

**Diagrama de transformações:**

```
Input: Imagem 28×28 grayscale
    ↓
[Conv1: kernel=5×5, filtros=10]
    ↓ Tamanho: 28-5+1 = 24×24
[ReLU + MaxPool 2×2]
    ↓ Tamanho: 24÷2 = 12×12
[Conv2: kernel=5×5, filtros=20]
    ↓ Tamanho: 12-5+1 = 8×8
[Dropout2D + ReLU + MaxPool 2×2]
    ↓ Tamanho: 8÷2 = 4×4
[Flatten: 20×4×4 = 320]
    ↓
[FC1: 320→50 + ReLU + Dropout]
    ↓
[FC2: 50→10 + LogSoftmax]
    ↓
Output: Log-probabilidades para 10 classes
```

---

## 2. Conceitos Fundamentais do PyTorch

### 2.1 Camadas Convolucionais (Conv2d)

**Definição**: Operação fundamental em visão computacional que aplica filtros deslizantes sobre a imagem para detectar características locais.

**Funcionamento**:
```python
nn.Conv2d(in_channels, out_channels, kernel_size)
```

**Parâmetros principais**:
- `in_channels`: Número de canais de entrada (1 para grayscale, 3 para RGB)
- `out_channels`: Número de filtros/características detectadas
- `kernel_size`: Tamanho da janela deslizante (ex: 3×3, 5×5)

**Como funciona**:
1. Um filtro (kernel) desliza sobre a imagem
2. Em cada posição, calcula produto escalar entre filtro e região da imagem
3. Gera um mapa de ativação mostrando onde a característica foi detectada

**Vantagens**:
- **Compartilhamento de parâmetros**: Mesmo filtro usado em toda imagem
- **Invariância espacial**: Detecta características independente da posição
- **Hierarquia**: Camadas profundas detectam padrões mais complexos

### 2.2 Regularização com Dropout

**Definição**: Técnica que aleatoriamente "desliga" neurônios durante treinamento para prevenir overfitting.

**Tipos no código**:

#### Dropout2d
```python
nn.Dropout2d(p=0.5)  # Zera 50% dos canais completos
```
- Aplicado em camadas convolucionais
- Remove mapas de características inteiros
- Mais efetivo que dropout pixel-wise em CNNs

#### Dropout1d (nas camadas densas)
```python
F.dropout(x, training=self.training)
```
- Aplicado em camadas lineares
- Remove neurônios individuais
- Apenas ativo durante `training=True`

**Por que funciona**:
1. **Redundância forçada**: Rede não pode depender de poucos neurônios
2. **Generalização**: Melhora performance em dados não vistos
3. **Ensemble implícito**: Simula múltiplas redes menores

**Analogia**: Como estudar sem depender sempre do mesmo colega - força aprendizado independente.

### 2.3 Camadas Lineares (Linear)

**Definição**: Transformação linear básica (multiplicação matriz + bias) usada para classificação final.

```python
nn.Linear(in_features, out_features)
```

**Operação matemática**:
```
y = xW^T + b
onde:
- x: entrada (batch_size, in_features)
- W: pesos treináveis (out_features, in_features)
- b: bias (out_features)
```

**Uso na arquitetura**:
- **fc1**: Combina características extraídas pelas convolucionais
- **fc2**: Mapeia para número de classes (10 dígitos)

**Transição espacial→conceitual**:
- Convolucionais: trabalham com informação espacial (2D)
- Lineares: trabalham com representações abstratas (1D)

### 2.4 Funções de Ativação

#### 2.4.1 ReLU

**Definição**: Rectified Linear Unit - função de ativação não-linear simples e eficaz.

**Fórmula**: `f(x) = max(0, x)`

**Comportamento**:
```
x < 0: f(x) = 0     (zera valores negativos)
x ≥ 0: f(x) = x     (mantém valores positivos)
```

**Vantagens**:
- **Computacionalmente eficiente**: Operação simples de comparação
- **Gradientes limpos**: Evita problema de desaparecimento de gradiente
- **Esparsidade**: Muitos neurônios ficam inativos (zero)

**Por que é importante**:
- Sem ativação não-linear, rede seria apenas regressão linear
- Permite aprender padrões complexos e não-lineares
- Introduz capacidade de "decisão" (ativa ou não ativa)

#### 2.4.2 Softmax e LogSoftmax

**Softmax**:
- Converte valores brutos em probabilidades que somam 1
- `softmax(x_i) = exp(x_i) / Σ exp(x_j)`

**LogSoftmax**:
- Logaritmo natural do softmax
- `log_softmax(x_i) = x_i - log(Σ exp(x_j))`

**Por que LogSoftmax**:
- **Estabilidade numérica**: Evita underflow/overflow
- **Eficiência**: Melhor para função de perda NLLLoss
- **Gradientes**: Computação mais estável durante backpropagation

### 2.5 Max Pooling

**Definição**: Operação de downsampling que reduz dimensões espaciais mantendo informações mais importantes.

**Funcionamento**:
```python
F.max_pool2d(x, kernel_size=2)
```

**Processo**:
1. Divide imagem em janelas não-sobrepostas (ex: 2×2)
2. Toma valor máximo de cada janela
3. Reduz dimensão pela metade

**Exemplo visual**:
```
Entrada 4×4:          Saída 2×2 (após max_pool 2×2):
[1  3  2  4]          [7  8]
[5  7  6  8]    →     [15 16]
[9  11 10 12]
[13 15 14 16]
```

**Benefícios**:
- **Redução computacional**: Menos parâmetros nas camadas seguintes  
- **Invariância à translação**: Pequenos deslocamentos não afetam resultado
- **Abstração**: Foca nas características mais "fortes"
- **Controle de overfitting**: Reduz complexidade do modelo

### 2.6 Normalização de Dados

**No contexto MNIST**:
```python
transforms.Normalize((0.1307,), (0.3081,))
```

**Fórmula**: `x_normalizado = (x - média) / desvio_padrão`

**Por que normalizar**:
- **Convergência mais rápida**: Gradientes mais estáveis
- **Escala uniforme**: Todas as features têm mesma importância inicial
- **Estabilidade numérica**: Evita valores muito grandes/pequenos

**Valores MNIST**:
- `média = 0.1307`: Valor médio dos pixels no dataset
- `desvio = 0.3081`: Dispersão padrão dos pixels
- **Resultado**: Pixels normalizados entre aproximadamente [-0.4, 2.8]

**Pipeline completo de pré-processamento**:
```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Garante 1 canal
    transforms.Resize((28, 28)),                  # Padroniza tamanho
    transforms.ToTensor(),                        # [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))    # Normalização final
])
```
