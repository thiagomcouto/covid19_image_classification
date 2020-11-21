# Utilizando image classification em imagens de raio-x para detecção de COVID-19

<p>A pandemia causada pelo Covid-19 atingiu todos os setores e com ela trouxe novos desafios, especialmente na área da saúde. Cientistas ao redor do mundo buscam novos tratamentos, formas de prevenção e detecção da doença. Nessa busca por novos métodos um requisito crucial é a escala, dadas as proporções da pandemia, se faz necessário que os resultados sejam escaláveis para alcançar uma grande parte da população.</p>

<p>Um dos métodos que podem ser utilizados para triagem de pacientes com suspeita do Covid-19 é a análise de raio-x e tomografia da região torácica, nessa análise podemos aplicar técnicas de classificação de imagem com Machine Learning para agilizar a detecção e escalar para um maior número de pacientes. Nessa linha algumas publicações como https://data.mendeley.com/datasets/8h65ywd2jr/3 exploram essa possibilidade.</p>

<p>Nesse blogpost vamos explorar a utilização do algoritmo built-in da AWS para classificação de imagem (https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html) para classificar imagens de raio-x torácico de pacientes entre covid-19 detectado ou não detectado. O algoritmo utiliza uma CNN(ResNet) e pode ser treinado utilizando transfer learning para melhores resultados quando um maior número de imagens não está disponível.</p> 

<p>A postagem foi inspirada nesse trabalho https://github.com/shervinmin/DeepCovid, bem como o dataset utilizado, que por sua vez é baseado nos datasets públicos https://github.com/ieee8023/covid-chestxray-dataset e https://stanfordmlgroup.github.io/competitions/chexpert/.

Para executar o notebook, utilizaremos um Sagemaker Notebook Instance com as configurações padrão, maiores detalhes de como criar nesse link: https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html </p>

## Preparando o Dataset

<p>Para iniciar vamos fazer o download do dataset https://github.com/shervinmin/DeepCovid/tree/master/data e organizar a estrutura das pasta


```python
! wget https://www.dropbox.com/s/09b5nutjxotmftm/data_upload_v2.zip
! unzip data_upload_v2.zip
```

    --2020-11-18 13:50:10--  https://www.dropbox.com/s/09b5nutjxotmftm/data_upload_v2.zip
    Resolving www.dropbox.com (www.dropbox.com)... 162.125.7.1, 2620:100:6050:1::a27d:b01
    Connecting to www.dropbox.com (www.dropbox.com)|162.125.7.1|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: /s/raw/09b5nutjxotmftm/data_upload_v2.zip [following]
    --2020-11-18 13:50:10--  https://www.dropbox.com/s/raw/09b5nutjxotmftm/data_upload_v2.zip
    Reusing existing connection to www.dropbox.com:443.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com/cd/0/inline/BDZ93C715GC0QFg08iEkD0uUe-MNw8-Y7SWw995ZYKK9fxZ6BcQxXIOEdWCEwxFjAtwaBByQJ0nc7lUVk4aBpBPOPacjGXV_b-hb841lg0xGsdwuuZNCeczgrDUl01ZJY9s/file# [following]
    --2020-11-18 13:50:11--  https://uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com/cd/0/inline/BDZ93C715GC0QFg08iEkD0uUe-MNw8-Y7SWw995ZYKK9fxZ6BcQxXIOEdWCEwxFjAtwaBByQJ0nc7lUVk4aBpBPOPacjGXV_b-hb841lg0xGsdwuuZNCeczgrDUl01ZJY9s/file
    Resolving uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com (uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com)... 162.125.11.15, 2620:100:6050:15::a27d:b0f
    Connecting to uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com (uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com)|162.125.11.15|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /cd/0/inline2/BDaCA06R4OJ-LGIV4YalvdDXtEIsnctL0avnDkak6eE-ONhmlaPl8O7JvPQFhX4WHbktnHrbH6vAIuplgd0HKS3KXLv9jUnbg6embbyxQb60y8hOxtIf1f5RL1UY-yuXyKjgo_LyzvwZPqtWfwqJVOT4FbJsv04NcM2uU0XSUf_2_qWVZHL94xy-Volod5BPaB6ZtwkgyrYjg2sh3_VWbMvlMKgiiEjYroieVrWdyj7sAefjUHwSnHasXOAuCwpwmIjKuKBzyBFlw4eUca-ve82Iz5jVbsrs6Ymkj78fwf2RBPOTvNDO6eKdKsh1MgKvqiuCih9efGvTNYXzq9DTBLvzCe8r8dcaD54aYrFIIvLbLw/file [following]
    --2020-11-18 13:50:12--  https://uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com/cd/0/inline2/BDaCA06R4OJ-LGIV4YalvdDXtEIsnctL0avnDkak6eE-ONhmlaPl8O7JvPQFhX4WHbktnHrbH6vAIuplgd0HKS3KXLv9jUnbg6embbyxQb60y8hOxtIf1f5RL1UY-yuXyKjgo_LyzvwZPqtWfwqJVOT4FbJsv04NcM2uU0XSUf_2_qWVZHL94xy-Volod5BPaB6ZtwkgyrYjg2sh3_VWbMvlMKgiiEjYroieVrWdyj7sAefjUHwSnHasXOAuCwpwmIjKuKBzyBFlw4eUca-ve82Iz5jVbsrs6Ymkj78fwf2RBPOTvNDO6eKdKsh1MgKvqiuCih9efGvTNYXzq9DTBLvzCe8r8dcaD54aYrFIIvLbLw/file
    Reusing existing connection to uc2d87e92467f170f35f4f6f2987.dl.dropboxusercontent.com:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 297183978 (283M) [application/zip]
    Saving to: ‘data_upload_v2.zip’
    
    data_upload_v2.zip  100%[===================>] 283.42M  53.8MB/s    in 5.4s    
    
    2020-11-18 13:50:18 (52.5 MB/s) - ‘data_upload_v2.zip’ saved [297183978/297183978]
    
    



```bash
%%bash
#Retirando as imagens das pastas de condições específicas e organizando as pastas

mv data_upload_v2/test/non/*/* data_upload_v2/test/non/
rm -rf data_upload_v2/test/non/*/

mkdir covid19_dataset
mkdir covid19_dataset/0_non/
mkdir covid19_dataset/1_covid/
mkdir test

mv data_upload_v2/test/non/* covid19_dataset/0_non/
mv data_upload_v2/train/non/* covid19_dataset/0_non/

mv data_upload_v2/test/covid/* covid19_dataset/1_covid/
mv data_upload_v2/train/covid/* covid19_dataset/1_covid/
```

O dataset atual possui imagens de outras enfermidades que para efeito dessa análise vamos considerar como "Covid não detectado", abaixo selecionamos algumas imagens do dataset para testar nosso modelo posteriormente, essas imagens não serão utilizadas no treinamento.


```bash
%%bash
# Sem Covid19
mv covid19_dataset/0_non/Atelectasis-patient35833-study1-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient00051-study1-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient00140-study4-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient01190-study2-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient04098-study5-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient01324-study4-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient01311-study3-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient05202-study5-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient11091-study1-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient16081-study20-view1_frontal.jpg test/
mv covid19_dataset/0_non/patient01772-study18-view1_frontal.jpg test/

#Com Covid19
mv covid19_dataset/1_covid/covid-19-pneumonia-22-day1-pa.png test/
mv covid19_dataset/1_covid/nejmoa2001191_f5-PA.jpeg test/
mv covid19_dataset/1_covid/5A78BCA9-5B7A-440D-8A4E-AE7710EA6EAD.jpeg test/
mv covid19_dataset/1_covid/radiol.2020201160.fig3a.jpeg test/



```

### Data Augmentation

O dataset atual está desbalanceado, contendo 184 imagens de pacientes diagnosticado com Covid-19 e 5 mil imagens de pacientes sem Covid-19. Para reduzir essa diferença vamos utilizar uma library em python para gerar 1 mil novas imagens de pacientes com Covid-19 positivo.


```python
#instalando a lib
! pip install Augmentor
```

    Requirement already satisfied: Augmentor in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (0.2.8)
    Requirement already satisfied: tqdm>=4.9.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from Augmentor) (4.42.1)
    Requirement already satisfied: future>=0.16.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from Augmentor) (0.18.2)
    Requirement already satisfied: numpy>=1.11.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from Augmentor) (1.18.1)
    Requirement already satisfied: Pillow>=5.2.0 in /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages (from Augmentor) (7.0.0)
    [33mWARNING: You are using pip version 20.0.2; however, version 20.2.4 is available.
    You should consider upgrading via the '/home/ec2-user/anaconda3/envs/mxnet_p36/bin/python -m pip install --upgrade pip' command.[0m



```python
#Utilizando rotacionamento e zoom para gerar 1 mil novos samples
import Augmentor
p = Augmentor.Pipeline("covid19_dataset/1_covid")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.sample(1000)

```

    Executing Pipeline:   0%|          | 0/1000 [00:00<?, ? Samples/s]

    Initialised with 180 image(s) found.
    Output directory set to covid19_dataset/1_covid/output.

    Processing <PIL.Image.Image image mode=RGB size=1723x1466 at 0x7F9A701ACE80>: 100%|██████████| 1000/1000 [03:35<00:00,  4.64 Samples/s]                 



```python
#Movendo as imagens geradas para a pasta do dataset

! mv covid19_dataset/1_covid/output/* covid19_dataset/1_covid/
! rm -R covid19_dataset/1_covid/output
```

### Gerando RecordIO

Com as imagens estruturadas nas pastas, vamos converter as imagens para MXNet RecordIO, formato recomendado, mais detalhes sobre o formato e os benefícios de utilizá-lo nesse link https://mxnet.apache.org/versions/1.7.0/api/architecture/note_data_loading.html


```python
# fazendo download do arquivo para conversão
import os
import urllib.request

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        
        
# Tool for creating lst file
download('https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py')
```

Separando o dataset com 80% dos dados para treinamento e 20% para validação e gerando os arquivos lst com as referências dos arquivos


```python
! python im2rec.py --list --recursive covid19 covid19_dataset --train-ratio=0.8
```

    0_non 0
    1_covid 1


Utilizando os arquivos lst criaremos os arquivos recordIO redimensionando as imagens para um tamanho único.


```bash
%%bash
python im2rec.py --resize 224 --num-thread 16 covid19_val covid19_dataset
python im2rec.py --resize 224 --num-thread 16 covid19_train covid19_dataset
```

    Creating .rec file from /home/ec2-user/SageMaker/covid19/covid19_val.lst in /home/ec2-user/SageMaker/covid19
    time: 0.05532670021057129  count: 0
    Creating .rec file from /home/ec2-user/SageMaker/covid19/covid19_train.lst in /home/ec2-user/SageMaker/covid19
    time: 0.014029741287231445  count: 0
    time: 16.85485863685608  count: 1000
    time: 14.63080620765686  count: 2000
    time: 10.580530166625977  count: 3000


## Criando o Modelo

Com o dataset criado, vamos iniciar a criação do modelo. Abaixo utilizamos o as libs boto3 e sagemaker para buscar a sessão e a role(provenientes do Notebook Instance), bem como a uri da imagem que vamos utilizar para treinamento.


```python
%%time
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri


role = get_execution_role()
sess = sagemaker.session.Session()
bucket = sess.default_bucket()

training_image = get_image_uri(boto3.Session().region_name, 'image-classification')
```

    'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.


    CPU times: user 963 ms, sys: 119 ms, total: 1.08 s
    Wall time: 4.4 s


### Upload do dataset

Com o dataset criado e divido em treino e validação, vamos utilizar da lib boto3 para fazermos o upload e armazenar em variáveis para serem usadas como channel posteriormente.


```python
def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3object = s3.Bucket(bucket).put_object(Key=key, Body=data)


s3_train_key = "covid19/train"
s3_validation_key = "covid19/validation"


s3_train = 's3://{}/{}/'.format(bucket, s3_train_key)
s3_validation = 's3://{}/{}/'.format(bucket, s3_validation_key)


upload_to_s3(s3_train_key, 'covid19_train.rec')
upload_to_s3(s3_validation_key, 'covid19_val.rec')

```

### Hyperparameters

Na célula abaixo informaremos os hyperparameters para o modelo, para essa postagem utilizaremos uma ResNet18 com transfer learning, 20 epochs e learning rate de 0.0005. Além disso vamos usar a funcionalidade nativa de data augmentation para aumentar o número de samples e configuração de early stop. A instância utilizada no treinamento é a ml.p2.xlarge. Demais configurações nos comentários.


```python
# Podemos utilizar alguns números de layers como 18, 34, 50, 101, 152 and 200
# Para esse modelo vamos utilizar 18
num_layers = "18" 
# Shape das imagens que vamos utilizar no treinamento
image_shape = "3,224,224"
# Utilizamos do arquivo lst para determinar quantos samples de treinamento temos
num_training_samples = sum(1 for line in open('covid19_train.lst'))
# O número de classes são 2, Detectado e Não Detectado
num_classes = "2"
# Vamos utilizar um batch size de 20
mini_batch_size =  "20"
# Para o estudo utilizaremos 20 epochs
epochs = "20"
# Testaremos o learning rate abaixo
learning_rate = "0.0005"
# Configuração para early stop para economia de tempo e custo
early_stop = "True"
# Tipo de técnica de augmentation utilizada
augmentation_type = "crop_color_transform"
```


```python
%%time
import time
import boto3
from time import gmtime, strftime


s3 = boto3.client('s3')
job_name_prefix = 'covid19-classification'
job_name = job_name_prefix + '-' + time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
training_params = \
{
    # specify the training docker image
    "AlgorithmSpecification": {
        "TrainingImage": training_image,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": 's3://{}/{}/output'.format(bucket, job_name_prefix)
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.p2.xlarge",
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        "image_shape": image_shape,
        "num_layers": str(num_layers),
        "num_training_samples": str(num_training_samples),
        "num_classes": str(num_classes),
        "mini_batch_size": str(mini_batch_size),
        "epochs": str(epochs),
        "learning_rate": str(learning_rate),
        "augmentation_type": augmentation_type,
        "use_pretrained_model": str(1),
        "early_stopping" : early_stop
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 360000
    },
# Abaixo inserimos os channels de treinamento e validação com os arquivos que fizemos upload anteriormente
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_train,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_validation,
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None"
        }
    ]
}
print('Training job name: {}'.format(job_name))
print('\nInput Data Location: {}'.format(training_params['InputDataConfig'][0]['DataSource']['S3DataSource']))
```

    Training job name: covid19-classification--2020-11-18-13-56-10
    
    Input Data Location: {'S3DataType': 'S3Prefix', 'S3Uri': 's3://sagemaker-us-east-1-709657544516/covid19/train/', 'S3DataDistributionType': 'FullyReplicated'}
    CPU times: user 8.85 ms, sys: 0 ns, total: 8.85 ms
    Wall time: 8.13 ms


### Treinando o Modelo

Com as configurações acima, vamos iniciar o treinamento do modelo e aguardar sua finalização.


```python
# Criar job de treinamento
sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**training_params)

# Validando que o job iniciou
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # Espera pelo término do treinamento e valida o status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))
```

    Training job current status: InProgress
    Training job ended with status: Completed


Na célula abaixo, vamos configurar um modelo baseado no resultado do treinamento anterior, em posse desse modelo podemos iniciar um endpoint para inferências em tempo real ou fazermos inferências em batch.


```python
%%time
import boto3
from time import gmtime, strftime

sage = boto3.Session().client(service_name='sagemaker') 

model_name="covid19-classification" + time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
print(model_name)
info = sage.describe_training_job(TrainingJobName=job_name)
model_data = info['ModelArtifacts']['S3ModelArtifacts']
print(model_data)

hosting_image = get_image_uri(boto3.Session().region_name, 'image-classification')

primary_container = {
    'Image': hosting_image,
    'ModelDataUrl': model_data,
}

create_model_response = sage.create_model(
    ModelName = model_name,
    ExecutionRoleArn = role,
    PrimaryContainer = primary_container)

print(create_model_response['ModelArn'])
```

    'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.


    covid19-classification-2020-11-18-14-12-44
    s3://sagemaker-us-east-1-709657544516/covid19-classification/output/covid19-classification--2020-11-18-13-56-10/output/model.tar.gz
    arn:aws:sagemaker:us-east-1:709657544516:model/covid19-classification-2020-11-18-14-12-44
    CPU times: user 102 ms, sys: 113 µs, total: 102 ms
    Wall time: 1.62 s


### Inferências em Batch

Com o modelo criado, vamos fazer upload dos dados de teste que separamos anteriormente e criarmos um job em batch para inferência. Como citado anteriormente, podemos também configurar um endpoint com o modelo e executarmos inferências em tempo real, mas para o objetivo desse blog post inferências em batch são suficientes. O job vai realizar as inferências das imagens especificadas no bucket S3 e armazenar o resultado em arquivos json na pasta output.


```python
# Upload dos dados de teste
s3_test = 's3://{}/covid19/test/'.format(bucket)

!aws s3 cp test $s3_test --recursive --quiet
```


```python
# Configurando o parâmetros para o batch transform
timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
batch_job_name="covid19-batch-transform" + timestamp
batch_input = s3_test
request = \
{
    "TransformJobName": batch_job_name,
    "ModelName": model_name,
    "MaxConcurrentTransforms": 10,
    "MaxPayloadInMB": 10,
    "BatchStrategy": "SingleRecord",
    "TransformOutput": {
        "S3OutputPath": 's3://{}/{}/output'.format(bucket, batch_job_name)
    },
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": s3_test
            }
        },
        "ContentType": "application/x-image",
        "SplitType": "None",
        "CompressionType": "None"
    },
    "TransformResources": {
            "InstanceType": "ml.p2.xlarge",
            "InstanceCount": 1
    }
}

print('Transform job name: {}'.format(batch_job_name))
print('\nInput Data Location: {}'.format(batch_input))
```

    Transform job name: covid19-batch-transform-2020-11-18-14-27-34
    
    Input Data Location: s3://sagemaker-us-east-1-709657544516/covid19/test/


Na célula abaixo criamos o job de inferência em batch e aguardamos a conclusão do mesmo


```python
sagemaker = boto3.client('sagemaker')
sagemaker.create_transform_job(**request)

print("Created Transform job with name: ", batch_job_name)

while(True):
    response = sagemaker.describe_transform_job(TransformJobName=batch_job_name)
    status = response['TransformJobStatus']
    if status == 'Completed':
        print("Transform job ended with status: " + status)
        break
    if status == 'Failed':
        message = response['FailureReason']
        print('Transform failed with the following error: {}'.format(message))
        raise Exception('Transform job failed') 
    time.sleep(30)  
```

    Created Transform job with name:  covid19-batch-transform-2020-11-18-14-27-34
    Transform job ended with status: Completed


## Validando o Modelo

Após a conclusão do job de inferência, vamos inspecionar os resultados na pasta output e validarmos como nosso modelo se saiu.


```python
from urllib.parse import urlparse
import json
import numpy as np

s3_client = boto3.client('s3')
prediction_categories = ['Não Detectado',"Detectado"]

def list_objects(s3_client, bucket, prefix):
    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    objects = [content['Key'] for content in response['Contents']]
    return objects

def get_label(s3_client, bucket, prefix):
    filename = prefix.split('/')[-1]
    s3_client.download_file(bucket, prefix, filename)
    with open(filename) as f:
        data = json.load(f)
        index = np.argmax(data['prediction'])
        probability = data['prediction'][index]
    print("Result: file - "+filename+" label - " + prediction_categories[index] + ", probability - " + str(probability))
    return prediction_categories[index], probability

inputs = list_objects(s3_client, bucket, urlparse(batch_input).path.lstrip('/'))

outputs = list_objects(s3_client, bucket, batch_job_name + "/output")

# Check prediction result of the first 2 images
[get_label(s3_client, bucket, prefix) for prefix in outputs]
```

    Result: file - 5A78BCA9-5B7A-440D-8A4E-AE7710EA6EAD.jpeg.out label - Detectado, probability - 0.9792001843452454
    Result: file - Atelectasis-patient35833-study1-view1_frontal.jpg.out label - Não Detectado, probability - 1.0
    Result: file - covid-19-pneumonia-22-day1-pa.png.out label - Detectado, probability - 0.932252049446106
    Result: file - nejmoa2001191_f5-PA.jpeg.out label - Detectado, probability - 0.9968767166137695
    Result: file - patient00051-study1-view1_frontal.jpg.out label - Não Detectado, probability - 1.0
    Result: file - patient00140-study4-view1_frontal.jpg.out label - Não Detectado, probability - 0.9999998807907104
    Result: file - patient01190-study2-view1_frontal.jpg.out label - Não Detectado, probability - 0.9999960660934448
    Result: file - patient01311-study3-view1_frontal.jpg.out label - Não Detectado, probability - 0.9999990463256836
    Result: file - patient01324-study4-view1_frontal.jpg.out label - Não Detectado, probability - 0.9999996423721313
    Result: file - patient01772-study18-view1_frontal.jpg.out label - Não Detectado, probability - 0.9999790191650391
    Result: file - patient04098-study5-view1_frontal.jpg.out label - Não Detectado, probability - 1.0
    Result: file - patient05202-study5-view1_frontal.jpg.out label - Não Detectado, probability - 1.0
    Result: file - patient11091-study1-view1_frontal.jpg.out label - Não Detectado, probability - 1.0
    Result: file - patient16081-study20-view1_frontal.jpg.out label - Não Detectado, probability - 0.9926134347915649
    Result: file - radiol.2020201160.fig3a.jpeg.out label - Detectado, probability - 0.9885025024414062





    [('Detectado', 0.9792001843452454),
     ('Não Detectado', 1.0),
     ('Detectado', 0.932252049446106),
     ('Detectado', 0.9968767166137695),
     ('Não Detectado', 1.0),
     ('Não Detectado', 0.9999998807907104),
     ('Não Detectado', 0.9999960660934448),
     ('Não Detectado', 0.9999990463256836),
     ('Não Detectado', 0.9999996423721313),
     ('Não Detectado', 0.9999790191650391),
     ('Não Detectado', 1.0),
     ('Não Detectado', 1.0),
     ('Não Detectado', 1.0),
     ('Não Detectado', 0.9926134347915649),
     ('Detectado', 0.9885025024414062)]



### Resultado e Próximos Passos

Devido ao caráter randômico da separação dos dados os resultados obtidos podem variar, nos meus testes realizados o modelo classificou todos os pacientes corretamente dentre as imagens de teste. Caso desejamos buscar melhores resultados podemos utilizar a feature do sagemaker de automatic model tunning, mais informações nesse link: https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html


```python

```
