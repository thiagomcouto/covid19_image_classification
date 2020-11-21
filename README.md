# Utilizando image classification e Augmented AI em imagens de raio-x para detecção de COVID-19 – Part 1

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
    
    Archive:  data_upload_v2.zip
       creating: data_upload_v2/test/
       creating: data_upload_v2/test/covid/
      inflating: data_upload_v2/test/covid/03BF7561-A9BA-4C3C-B8A0-D3E585F73F3C.jpeg  
      inflating: data_upload_v2/test/covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png  
      inflating: data_upload_v2/test/covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-001-fig2b.png  
      inflating: data_upload_v2/test/covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-002-fig3a.png  
      inflating: data_upload_v2/test/covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-002-fig3b.png  
      inflating: data_upload_v2/test/covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-003-fig4a.png  
      inflating: data_upload_v2/test/covid/1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-003-fig4b.png  
      inflating: data_upload_v2/test/covid/1312A392-67A3-4EBF-9319-810CF6DA5EF6.jpeg  
      inflating: data_upload_v2/test/covid/16654_1_1.png  
      inflating: data_upload_v2/test/covid/16654_2_1.jpg  
      inflating: data_upload_v2/test/covid/16654_4_1.jpg  
      inflating: data_upload_v2/test/covid/16660_1_1.jpg  
      inflating: data_upload_v2/test/covid/16660_2_1.jpg  
      inflating: data_upload_v2/test/covid/16660_3_1.jpg  
      inflating: data_upload_v2/test/covid/16660_4_1.jpg  
      inflating: data_upload_v2/test/covid/16663_1_1.jpg  
      inflating: data_upload_v2/test/covid/16669_1_1.jpeg  
      inflating: data_upload_v2/test/covid/16669_3_1.jpeg  
      inflating: data_upload_v2/test/covid/16672_1_1.jpg  
      inflating: data_upload_v2/test/covid/16674_1_1.jpg  
      inflating: data_upload_v2/test/covid/1-s2.0-S0140673620303706-fx1_lrg.jpg  
      inflating: data_upload_v2/test/covid/1-s2.0-S0929664620300449-gr2_lrg-a.jpg  
      inflating: data_upload_v2/test/covid/1-s2.0-S0929664620300449-gr2_lrg-b.jpg  
      inflating: data_upload_v2/test/covid/1-s2.0-S0929664620300449-gr2_lrg-c.jpg  
      inflating: data_upload_v2/test/covid/1-s2.0-S0929664620300449-gr2_lrg-d.jpg  
      inflating: data_upload_v2/test/covid/1-s2.0-S1684118220300682-main.pdf-003-b1.png  
      inflating: data_upload_v2/test/covid/1-s2.0-S1684118220300682-main.pdf-003-b2.png  
      inflating: data_upload_v2/test/covid/41591_2020_819_Fig1_HTML.webp-day10.png  
      inflating: data_upload_v2/test/covid/41591_2020_819_Fig1_HTML.webp-day5.png  
      inflating: data_upload_v2/test/covid/446B2CB6-B572-40AB-B01F-1910CA07086A.jpeg  
      inflating: data_upload_v2/test/covid/4ad30bc6-2da0-4f84-bc9b-62acabfd518a.annot.original.png  
      inflating: data_upload_v2/test/covid/53EC07C9-5CC6-4BE4-9B6F-D7B0D72AAA7E.jpeg  
      inflating: data_upload_v2/test/covid/58cb9263f16e94305c730685358e4e_jumbo.jpeg  
      inflating: data_upload_v2/test/covid/5CBC2E94-D358-401E-8928-965CCD965C5C.jpeg  
      inflating: data_upload_v2/test/covid/6b3bdbc31f65230b8cdcc3cef5f8ba8a-40ac-0.jpg  
      inflating: data_upload_v2/test/covid/6b44464d-73a7-4cf3-bbb6-ffe7168300e3.annot.original.jpeg  
      inflating: data_upload_v2/test/covid/6C94A287-C059-46A0-8600-AFB95F4727B7.jpeg  
      inflating: data_upload_v2/test/covid/7AF6C1AF-D249-4BD2-8C26-449304105D03.jpeg  
      inflating: data_upload_v2/test/covid/7-fatal-covid19.jpg  
      inflating: data_upload_v2/test/covid/88de9d8c39e946abd495b37cd07d89e5-0666-0.jpg  
      inflating: data_upload_v2/test/covid/88de9d8c39e946abd495b37cd07d89e5-2ee6-0.jpg  
      inflating: data_upload_v2/test/covid/88de9d8c39e946abd495b37cd07d89e5-6531-0.jpg  
      inflating: data_upload_v2/test/covid/9fdd3c3032296fd04d2cad5d9070d4_jumbo.jpeg  
      inflating: data_upload_v2/test/covid/A7E260CE-8A00-4C5F-A7F5-27336527A981.jpeg  
      inflating: data_upload_v2/test/covid/ajr.20.23034.pdf-001.png  
      inflating: data_upload_v2/test/covid/ajr.20.23034.pdf-003.png  
      inflating: data_upload_v2/test/covid/AR-1.jpg  
      inflating: data_upload_v2/test/covid/AR-2.jpg  
      inflating: data_upload_v2/test/covid/auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg  
      inflating: data_upload_v2/test/covid/auntminnie-b-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg  
      inflating: data_upload_v2/test/covid/auntminnie-c-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg  
      inflating: data_upload_v2/test/covid/auntminnie-d-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg  
      inflating: data_upload_v2/test/covid/B2D20576-00B7-4519-A415-72DE29C90C34.jpeg  
      inflating: data_upload_v2/test/covid/CD50BA96-6982-4C80-AE7B-5F67ACDBFA56.jpeg  
      inflating: data_upload_v2/test/covid/covid-19-infection-exclusive-gastrointestinal-symptoms-pa.png  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-12.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-14-PA.png  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-20.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-20-pa-on-admission.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-23-day1.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-23-day3.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-23-day9.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-24-day12.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-24-day6.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-24-day7.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-28.png  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-30-PA.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-42.jpeg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-49-day4.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-49-day8.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-58-day-10.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-58-day-7.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-58-day-9.jpg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-67.jpeg  
      inflating: data_upload_v2/test/covid/covid-19-pneumonia-bilateral.jpg  
      inflating: data_upload_v2/test/covid/D7AF463C-2369-492D-908D-BE1911CCD74C.jpeg  
      inflating: data_upload_v2/test/covid/da9e9aac-de8c-44c7-ba57-e7cc8e4caaba.annot.original.jpeg  
      inflating: data_upload_v2/test/covid/E1724330-1866-4581-8CD8-CEC9B8AFEDDE.jpeg  
      inflating: data_upload_v2/test/covid/F63AB6CE-1968-4154-A70F-913AF154F53D.jpeg  
      inflating: data_upload_v2/test/covid/fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original.png  
      inflating: data_upload_v2/test/covid/figure1-5e73d7ae897e27ff066a30cb-98.jpeg  
      inflating: data_upload_v2/test/covid/jkms-35-e79-g001-l-b.jpg  
      inflating: data_upload_v2/test/covid/jkms-35-e79-g001-l-c.jpg  
      inflating: data_upload_v2/test/covid/kjr-21-e24-g003-l-a.jpg  
      inflating: data_upload_v2/test/covid/nejmc2001573_f1a.jpeg  
      inflating: data_upload_v2/test/covid/nejmc2001573_f1b.jpeg  
      inflating: data_upload_v2/test/covid/post-intubuation-pneumomediastium-and-pneumothorax-background-covid-19-pneumonia-day1.jpg  
      inflating: data_upload_v2/test/covid/post-intubuation-pneumomediastium-and-pneumothorax-background-covid-19-pneumonia-day6-1.jpg  
      inflating: data_upload_v2/test/covid/post-intubuation-pneumomediastium-and-pneumothorax-background-covid-19-pneumonia-day6-2.jpg  
      inflating: data_upload_v2/test/covid/post-intubuation-pneumomediastium-and-pneumothorax-background-covid-19-pneumonia-day7.jpg  
      inflating: data_upload_v2/test/covid/radiol.2020200490.fig3.jpeg  
      inflating: data_upload_v2/test/covid/radiol.2020201160.fig2a.jpeg  
      inflating: data_upload_v2/test/covid/radiol.2020201160.fig2c.jpeg  
      inflating: data_upload_v2/test/covid/ryct.2020200028.fig1a.jpeg  
      inflating: data_upload_v2/test/covid/ryct.2020200034.fig5-day0.jpeg  
      inflating: data_upload_v2/test/covid/ryct.2020200034.fig5-day4.jpeg  
      inflating: data_upload_v2/test/covid/ryct.2020200034.fig5-day7.jpeg  
      inflating: data_upload_v2/test/covid/yxppt-2020-02-19_00-51-27_287214-day10.jpg  
      inflating: data_upload_v2/test/covid/yxppt-2020-02-19_00-51-27_287214-day12.jpg  
      inflating: data_upload_v2/test/covid/yxppt-2020-02-19_00-51-27_287214-day8.jpg  
       creating: data_upload_v2/test/non/
       creating: data_upload_v2/test/non/Atelectasis/
      inflating: data_upload_v2/test/non/Atelectasis/patient00024-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00109-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00150-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00224-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00262-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00280-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00310-study29-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00312-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00317-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00344-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00348-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00361-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00376-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00572-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00593-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00602-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00690-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00717-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00721-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00724-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00728-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00847-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00888-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00928-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00950-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00967-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00972-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient00994-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01051-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01192-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01204-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01271-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01324-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01324-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01340-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01390-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01436-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01502-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01608-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01637-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01653-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01685-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01701-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01722-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01744-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01749-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient01951-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02050-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02067-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02082-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02137-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02137-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02201-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02239-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02271-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02313-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02332-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02332-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02416-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02424-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02443-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02482-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02483-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02531-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02548-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02560-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02592-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02592-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02715-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02720-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02727-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02747-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02780-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02851-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02903-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02975-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient02977-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03065-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03072-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03095-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03108-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03108-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03156-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03357-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03364-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03428-study25-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03483-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03759-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03883-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient03985-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04015-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04018-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04079-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04116-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04151-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04153-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04164-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04209-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04316-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04347-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04382-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04437-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04466-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04689-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04733-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04773-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04773-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04826-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04918-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04952-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04974-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient04989-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient05000-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient05008-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient05017-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Atelectasis/patient05031-study2-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Cardiomegaly/
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00111-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00145-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00237-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00271-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00294-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00315-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00339-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00340-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00356-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00392-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00392-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00481-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00528-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00547-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00574-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00684-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00693-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00762-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00785-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00826-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00826-study26-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00837-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00879-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00887-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00888-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient00991-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01019-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01019-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01052-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01064-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01103-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01167-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01167-study48-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01190-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01214-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01214-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01276-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01283-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01311-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01349-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01349-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01423-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01470-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01491-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01622-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01630-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01695-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01750-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01772-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01772-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01772-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01798-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01897-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01963-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient01993-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02121-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02176-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02198-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02242-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02242-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02292-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02312-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02315-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02315-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02315-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02375-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02418-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02431-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02433-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02716-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02826-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02860-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient02953-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03027-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03027-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03027-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03027-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03033-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03107-study21-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03133-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03238-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03238-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03320-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03325-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03441-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03486-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03594-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03615-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03844-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03936-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient03973-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04022-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04045-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04084-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04089-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04092-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04098-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04098-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04117-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04308-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04308-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04332-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04338-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04362-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04362-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04362-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04499-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04516-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04550-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04626-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04817-study24-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Cardiomegaly/patient04822-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Consolidation/
      inflating: data_upload_v2/test/non/Consolidation/patient00294-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient00385-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient00410-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient00410-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient00424-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient00662-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01069-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01071-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01137-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01248-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01374-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01446-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01487-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01634-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01708-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01832-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01914-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01922-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01931-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient01966-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02014-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02074-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02079-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02194-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02195-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02263-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02342-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02343-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02524-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02590-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02722-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient02760-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03044-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03044-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03219-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03219-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03230-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03254-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03461-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03504-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03562-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03619-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03799-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03799-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03799-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03887-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient03956-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04033-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04033-study2-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04148-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04195-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04261-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04283-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04344-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04394-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04399-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04399-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04421-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04532-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04573-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04613-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04713-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04924-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient04977-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05047-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05126-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05165-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05165-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05165-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05330-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05621-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05655-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05745-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05745-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05792-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05934-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05959-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient05987-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06021-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06070-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06151-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06190-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06357-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06398-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06495-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06501-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06563-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06591-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06604-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06653-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06883-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient06971-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07055-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07055-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07075-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07109-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07396-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07441-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07441-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07763-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07849-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient07867-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08058-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08299-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08337-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08382-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08713-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08779-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08950-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08963-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient08963-study10-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Consolidation/patient09052-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Edema/
      inflating: data_upload_v2/test/non/Edema/patient00003-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00015-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00098-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00100-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00114-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00115-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00222-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00246-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00342-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00349-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00359-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00392-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00468-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00500-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00532-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00534-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00542-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00562-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00595-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00693-study23-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00693-study25-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00693-study26-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00714-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00785-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00809-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00813-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00822-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00827-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00863-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00877-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00877-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00877-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00920-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00936-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00936-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00947-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00952-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient00984-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01013-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01032-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01110-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01113-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01123-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01129-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01155-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01167-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01191-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01207-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01221-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01221-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01221-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01221-study23-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01286-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01286-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01291-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01297-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01311-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01313-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01323-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01402-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01423-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01447-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01475-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01475-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01480-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01496-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01541-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01541-study3-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01541-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01545-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01577-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01579-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01582-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01627-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01682-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01767-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01772-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01776-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01785-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01797-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01805-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01838-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01891-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01893-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01918-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01918-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01932-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01932-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient01987-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02001-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02076-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02084-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02124-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02145-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02146-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02146-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02175-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02177-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02195-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02234-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02267-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02276-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02308-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02359-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02387-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02391-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02392-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02397-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02397-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02403-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02403-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02437-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02451-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02453-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02459-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02466-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02483-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02502-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02502-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Edema/patient02508-study2-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00121-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00285-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00307-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00327-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00368-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00475-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00582-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00638-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00650-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00687-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00760-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00869-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00879-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient00892-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01023-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01149-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01446-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01475-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01571-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01608-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01682-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01727-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01818-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01850-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01855-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01894-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient01927-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02058-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02093-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02096-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02311-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02393-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02543-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02550-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02603-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02622-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02701-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02734-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02734-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02741-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02741-study7-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02794-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02810-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02887-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient02944-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03054-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03079-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03149-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03155-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03428-study28-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03554-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03747-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03780-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03781-study22-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03835-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03853-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03946-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03960-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient03966-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04066-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04092-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04136-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04141-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04211-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04235-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04385-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04412-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04412-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04426-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04428-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04465-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04585-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04596-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04616-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04649-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04681-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04823-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04846-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04847-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04881-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04891-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient04911-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05086-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05163-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05202-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05222-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05245-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05384-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05508-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05575-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05609-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05647-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05649-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05767-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05785-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient05947-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06101-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06303-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06349-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06354-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06494-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06609-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06845-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient06877-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07113-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07115-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07184-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07271-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07271-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07336-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07337-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07519-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07521-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Enlarged_Cardiomediastinum/patient07545-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Fracture/
      inflating: data_upload_v2/test/non/Fracture/patient00051-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00109-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00109-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00127-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00294-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00300-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00343-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00365-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00441-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00495-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00530-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00536-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00626-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00674-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00736-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00774-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00787-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00826-study25-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient00924-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01139-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01140-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01190-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01288-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01343-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01427-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01677-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01726-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01765-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01768-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01771-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01784-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01887-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01908-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient01982-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02075-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02109-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02164-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02203-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02404-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02438-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02462-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02462-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02495-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02504-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02525-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02549-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02582-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02585-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02630-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02788-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02852-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02857-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02861-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02876-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02876-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02911-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02957-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient02985-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03011-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03038-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03038-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03096-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03225-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03225-study5-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03225-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03262-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03291-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03299-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03310-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03385-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03419-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03451-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03464-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03500-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03519-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03565-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03598-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03670-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03695-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03709-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03732-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03795-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03836-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03847-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03864-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient03936-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04002-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04092-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04104-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04133-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04166-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04214-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04218-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04231-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04254-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04317-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04317-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04323-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04331-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04419-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04419-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04436-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04515-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04530-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04575-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04610-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04687-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04687-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04704-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04765-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04765-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04838-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04873-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04895-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04901-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient04987-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient05054-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient05065-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Fracture/patient05072-study2-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Lung_Lesion/
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00179-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00179-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00255-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00395-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00413-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00418-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00551-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00602-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient00639-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01094-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01282-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01282-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01306-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01395-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01463-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01484-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01484-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01490-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01823-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01826-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient01857-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02041-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02049-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02179-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02349-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02360-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02360-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02643-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02643-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02652-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02702-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02857-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02888-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient02888-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient03084-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient03206-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient03534-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient03660-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient03977-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient03983-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient04297-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient04401-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient04504-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient04731-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient04815-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient04915-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05094-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05152-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05167-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05268-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05324-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05335-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05483-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05539-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05676-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05706-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05742-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05793-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient05933-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06044-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06168-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06295-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06319-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06352-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06444-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06483-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06684-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06747-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06757-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient06958-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient07252-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient07536-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient07601-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient07618-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient07741-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient07933-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08170-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08266-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08338-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08653-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08679-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08752-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08792-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient08896-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09061-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09147-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09442-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09750-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09845-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09848-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09895-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient09934-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10029-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10029-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10167-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10278-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10283-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10349-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10415-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10557-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10597-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10641-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10660-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10678-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10703-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10711-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10892-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient10934-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Lesion/patient11091-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Lung_Opacity/
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00034-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00035-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00036-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00046-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00076-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00092-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00112-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00115-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00135-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00140-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00140-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00140-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00140-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00141-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00155-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00184-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00189-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00201-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00204-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00204-study7-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00204-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00210-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00210-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00210-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00211-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00213-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00215-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00235-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00270-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00275-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00280-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00290-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00299-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00314-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00314-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00323-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00326-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00326-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00386-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00388-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00391-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00416-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00440-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00447-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00451-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00464-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00465-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00467-study45-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00490-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00497-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00501-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00510-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00522-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00522-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00528-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00534-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00534-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00539-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00556-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00559-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00559-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00560-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00569-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00611-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00612-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00639-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00651-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00657-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00669-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00670-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00691-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00696-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00716-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00718-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00719-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00734-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00741-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00750-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00754-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00755-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00761-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00792-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00805-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00850-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00857-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00877-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00877-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00877-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00880-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00910-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00923-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00929-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00931-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00932-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00936-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00936-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00937-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00944-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00948-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00953-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00955-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00979-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00980-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00996-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient00997-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01002-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01003-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01008-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01030-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01032-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01035-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01051-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01051-study19-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01066-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01066-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01068-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01069-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Lung_Opacity/patient01072-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/No_Finding/
      inflating: data_upload_v2/test/non/No_Finding/patient00004-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00006-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00010-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00013-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00021-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00025-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00032-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00049-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00054-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00057-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00057-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00060-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00066-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00070-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00071-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00077-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00079-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00082-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00084-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00097-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00106-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00107-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00110-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00111-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00120-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00123-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00123-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00126-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00137-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00140-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00152-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00154-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00163-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00175-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00184-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00202-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00202-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00203-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00205-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00211-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00216-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00217-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00218-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00223-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00226-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00230-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00234-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00239-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00242-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00243-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00253-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00254-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00259-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00262-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00265-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00272-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00277-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00281-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00284-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00304-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00310-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00312-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00322-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00327-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00331-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00332-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00333-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00348-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00349-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00352-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00353-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00357-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00360-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00387-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00390-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00398-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00400-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00402-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00410-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00410-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00419-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00428-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00432-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00438-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00446-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00449-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00450-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00466-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00466-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00466-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00467-study28-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00483-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00484-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00486-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00488-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00488-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00493-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00494-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00499-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00500-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00505-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00518-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00528-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00531-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00557-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00558-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00567-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00567-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00577-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00578-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00590-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00595-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00599-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00602-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00606-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00608-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00614-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00615-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00622-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00631-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00631-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00633-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00635-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00636-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00640-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00656-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00667-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00672-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00676-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00677-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00692-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00698-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00699-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00700-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00701-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00728-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00732-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00743-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00751-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00754-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00755-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00763-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00765-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00782-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00788-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00800-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00801-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00807-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00810-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00812-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00817-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00821-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00829-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00832-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00839-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00841-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00844-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00851-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00856-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00857-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00859-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00866-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00876-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00881-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00900-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00901-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00903-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00906-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00908-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00909-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00911-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00916-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00917-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00957-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00961-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00973-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00980-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00986-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient00989-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01001-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01007-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01020-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01025-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01040-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01047-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01049-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01053-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01059-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01065-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01074-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01075-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01076-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01078-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01083-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01086-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01087-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01088-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01095-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01103-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01104-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01105-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01106-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01120-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01122-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01134-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01135-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01153-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01158-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01167-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01169-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01171-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01189-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01211-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01221-study25-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01224-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01227-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01228-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01232-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01279-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01285-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01300-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01301-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01307-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01307-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01318-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01319-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01329-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01334-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01335-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01335-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01342-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01349-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01353-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01358-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01360-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01364-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01368-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01382-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01386-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01391-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01400-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01408-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01413-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01431-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01432-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01433-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01434-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01439-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01444-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01452-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01454-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01454-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01456-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01458-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01459-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01478-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01482-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01495-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01498-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01498-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01504-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01509-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01514-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01514-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01516-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01516-study8-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01517-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01528-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01531-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01535-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01536-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01544-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01548-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01552-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01553-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01557-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01559-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01564-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01575-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01584-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01585-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01592-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01598-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01601-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01608-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01615-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01618-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01622-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01625-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01642-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01643-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01652-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01669-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01672-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01681-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01686-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01691-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01696-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01705-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01711-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01712-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01732-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01734-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01739-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01743-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01750-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01754-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01758-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01786-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01790-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01798-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01801-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01802-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01817-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01818-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01822-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01825-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01825-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01835-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01839-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01840-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01842-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01843-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01853-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01860-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01863-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01868-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01871-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01880-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01883-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01890-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01902-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01904-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01904-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01907-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01915-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01915-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01915-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01917-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01928-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01934-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01938-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01940-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01948-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01971-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01979-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01982-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01984-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01985-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01989-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient01994-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02000-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02002-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02005-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02025-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02029-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02037-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02051-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02069-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02072-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02075-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02082-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02085-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02086-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02089-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02089-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02090-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02091-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02106-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02108-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02116-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02117-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02118-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02119-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02120-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02135-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02141-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02165-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02174-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02189-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02199-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02203-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02203-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02207-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02207-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02209-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02216-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02219-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02220-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02225-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02229-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02230-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02232-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02233-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02238-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02240-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02248-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02251-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02253-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02269-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02272-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02281-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02287-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02288-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02289-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02294-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02296-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02327-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02328-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02330-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02333-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02337-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02344-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02345-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02346-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02360-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02370-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02372-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02376-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02378-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02378-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02380-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02382-study28-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02385-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02386-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02390-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02394-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02414-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02419-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02420-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02427-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02427-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02434-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02435-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02452-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02455-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02462-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02462-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02464-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02467-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02473-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02476-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02478-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02479-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02483-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02483-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02487-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02491-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02492-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02500-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02503-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02505-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02507-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02509-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02510-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02513-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02527-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02554-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02555-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02559-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02564-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02566-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02566-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02570-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02601-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02602-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02602-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02609-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02611-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02612-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02612-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02617-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02619-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02625-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02638-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02644-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02649-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02650-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02654-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02656-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02659-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02660-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02660-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02686-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02716-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02718-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02718-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02724-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02730-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02740-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02746-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02758-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02769-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02778-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02784-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02786-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02802-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02814-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02827-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02838-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02844-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02846-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02849-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02854-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02856-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02862-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02878-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02880-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02880-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02881-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02884-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02884-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02884-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02885-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02896-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02899-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02908-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02914-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02932-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02937-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02938-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02943-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02946-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02949-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02956-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02959-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02974-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient02980-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03000-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03018-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03022-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03022-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03035-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03035-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03039-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03040-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03042-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03045-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03047-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03048-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03048-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03072-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03073-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03073-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03081-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03082-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03085-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03088-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03097-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03100-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03102-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03108-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03109-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03113-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03135-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03143-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03150-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03172-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03174-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03175-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03178-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03183-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03185-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03205-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03212-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03221-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03226-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03230-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03233-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03237-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03243-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03251-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03253-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03259-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03271-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03287-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03294-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03296-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03302-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03304-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03304-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03305-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03305-study19-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03308-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03315-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03318-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03322-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03327-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03332-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03334-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03359-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03360-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03361-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03366-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03370-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03370-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03375-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03377-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03382-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03386-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03386-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03394-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03397-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03401-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03405-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03410-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03410-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03422-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03422-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03443-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03448-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03452-study23-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03452-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03453-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03453-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03456-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03469-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03472-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03474-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03479-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03488-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03488-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03490-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03495-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03497-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03499-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03499-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03504-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03505-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03516-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03524-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03531-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03531-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03543-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03551-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03558-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03561-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03569-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03579-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03579-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03579-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03579-study4-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03582-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03583-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03587-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03600-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03602-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03612-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03623-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03630-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03645-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03646-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03651-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03653-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03655-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03655-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03656-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03658-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03672-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03672-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03674-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03678-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03684-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03694-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03717-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03719-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03727-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03730-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03731-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03744-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03749-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03751-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03751-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03752-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03753-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03757-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03760-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03760-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03761-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03761-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03771-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03772-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03775-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03796-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03797-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03804-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03805-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03805-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03815-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03816-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03818-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03819-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03820-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03824-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03826-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03827-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03844-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03845-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03846-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03854-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03855-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03857-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03860-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03867-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03868-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03878-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03889-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03895-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03907-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03911-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03920-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03922-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03922-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03925-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03928-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03933-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03939-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03940-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03947-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03949-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03950-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03953-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03959-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03970-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03972-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03972-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03974-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03976-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient03995-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04003-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04004-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04005-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04016-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04018-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04024-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04031-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04036-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04040-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04042-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04044-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04056-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04066-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04067-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04067-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04072-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04079-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04081-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04086-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04087-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04097-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04098-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04098-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04100-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04106-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04107-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04125-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04128-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04144-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04153-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04156-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04159-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04161-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04168-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04174-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04180-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04193-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04194-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04197-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04213-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04216-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04220-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04222-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04226-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04231-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04236-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04246-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04247-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04257-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04262-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04270-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04271-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04271-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04290-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04290-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04296-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04301-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04306-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04310-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04311-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04319-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04321-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04322-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04329-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04334-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04345-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04356-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04362-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04371-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04374-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04385-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04385-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04393-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04396-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04402-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04403-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04404-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04423-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04426-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04429-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04433-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04438-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04438-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04447-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04454-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04467-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04467-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04483-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04493-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04500-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04507-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04509-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04510-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04512-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04518-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04533-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04543-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04544-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04552-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04562-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04563-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04570-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04586-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04593-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04594-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04597-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04604-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04604-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04611-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04612-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04617-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04618-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04624-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04631-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04631-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04634-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04640-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04642-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04644-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04651-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04662-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04669-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04671-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04672-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04675-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04686-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04690-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04695-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04696-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04697-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04697-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04702-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04702-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04706-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04719-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04723-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04734-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04742-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04746-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04751-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04752-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04764-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04773-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04802-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04808-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04817-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04817-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04821-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04827-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04828-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04835-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04842-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04844-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04853-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04859-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04861-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04863-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04866-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04870-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04881-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04881-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04881-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04881-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04890-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04892-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04895-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04905-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04933-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04942-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04947-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04950-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04951-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04953-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04955-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04970-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04970-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04972-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04979-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04997-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04998-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient04999-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05002-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05005-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05016-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05020-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05030-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05033-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05034-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05046-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05064-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05069-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05076-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05089-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05092-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05105-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05129-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05133-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05150-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05155-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05157-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05161-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05163-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05167-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05168-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05170-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05173-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05175-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05176-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05179-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05180-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05184-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05199-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05208-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05209-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05210-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05215-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05235-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05235-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05238-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05249-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05256-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05261-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05275-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05281-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05288-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05290-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05303-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05316-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05322-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05331-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05334-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05343-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05345-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05346-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05349-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05349-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05349-study18-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05350-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05352-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05353-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05358-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05358-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05368-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05371-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05376-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05386-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05387-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05395-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05399-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05400-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05402-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05411-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05413-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05426-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05430-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05434-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05439-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05439-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05448-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05451-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05453-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05453-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05459-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05465-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05469-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05471-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05472-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05481-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05483-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05490-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05494-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05499-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05501-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05502-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05503-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05511-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05518-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05523-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05526-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05533-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05535-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05541-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05547-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05556-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05564-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05566-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05569-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05573-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05576-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05577-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05581-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05586-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05592-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05594-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05597-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05600-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05603-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05608-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05610-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05619-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05619-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05619-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05620-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05623-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05624-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05632-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05633-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05640-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05640-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05641-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05674-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05675-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05675-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05675-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05678-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05678-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05678-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05681-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05689-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05694-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05696-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05700-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05703-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05707-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05708-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05716-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05717-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05719-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05725-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05744-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05756-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05770-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05787-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05804-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05811-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05822-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05822-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05826-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05834-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05838-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05840-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05842-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05845-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05854-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05864-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05865-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05869-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05879-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05883-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05896-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05910-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05919-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05921-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05926-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05930-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05933-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05950-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05952-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05959-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05963-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05966-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05966-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05971-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05980-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05989-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05995-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05995-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient05995-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06005-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06008-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06008-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06013-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06015-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06018-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06019-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06045-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06048-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06049-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06052-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06076-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06087-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06088-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06090-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06094-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06098-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06102-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06104-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06108-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06117-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06124-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06131-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06132-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06142-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06144-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06145-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06147-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06153-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06160-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06178-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06181-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06188-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06191-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06206-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06218-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06228-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06239-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06239-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06242-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06258-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06260-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06264-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06267-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06276-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06284-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06285-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06288-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06309-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06315-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06325-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06328-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06342-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06346-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06346-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06346-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06364-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06369-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06379-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06384-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06386-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06398-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06431-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06435-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06436-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06440-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06441-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06445-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06457-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06464-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06464-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06471-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06472-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06492-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06506-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06520-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06523-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06533-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06537-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06539-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06555-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06566-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06569-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06574-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06575-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06577-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06579-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06594-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06598-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06607-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06621-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06635-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06639-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06642-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06645-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06646-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06651-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06658-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06671-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06672-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06677-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06677-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06687-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06699-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06702-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06707-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06716-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06723-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06725-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06727-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06732-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06740-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06745-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06745-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06752-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06753-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06760-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06760-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06768-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06771-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06771-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06778-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06782-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06788-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06797-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06801-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06802-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06817-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06820-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06820-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06821-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06825-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06828-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06841-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06843-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06843-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06845-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06845-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06857-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06860-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06871-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06875-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06877-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06879-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06883-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06885-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06908-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06921-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06923-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06928-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06929-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06932-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06945-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06958-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06967-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06973-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06974-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06974-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06976-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06981-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06982-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient06997-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07016-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07022-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07025-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07031-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07032-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07040-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07050-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07056-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07071-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07071-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07076-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07077-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07077-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07079-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07082-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07096-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07097-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07101-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07107-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07108-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07112-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07113-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07131-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07133-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07151-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07160-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07165-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07165-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07169-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07172-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07173-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07179-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07181-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07192-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07193-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07195-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07196-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07201-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07202-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07203-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07208-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07208-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07212-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07213-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07213-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07214-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07214-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07215-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07216-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07220-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07223-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07233-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07242-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07243-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07254-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07260-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07263-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07267-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07268-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07275-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07282-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07287-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07290-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07290-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07293-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07299-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07305-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07316-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07318-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07323-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07332-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07333-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07334-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07352-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07356-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07374-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07377-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07378-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07378-study7-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07392-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07394-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07394-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07399-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07405-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07410-study30-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07414-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07418-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07423-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07424-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07430-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07432-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07438-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07447-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07451-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07459-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07462-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07476-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07477-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07484-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07485-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07487-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07490-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07498-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07499-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07502-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07524-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07530-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07549-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07549-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07555-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07557-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07574-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07574-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07575-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07577-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07583-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07593-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07614-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07614-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07621-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07624-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07632-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07632-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07635-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07651-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07651-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07651-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07652-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07655-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07662-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07662-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07663-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07667-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07677-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07683-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07688-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07690-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07699-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07700-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07730-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07730-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07733-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07739-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07740-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07740-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07744-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07748-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07763-study30-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07764-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07764-study9-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07770-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07777-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07779-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07780-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07780-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07781-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07782-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07790-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07792-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07794-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07796-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07803-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07819-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07819-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07825-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07834-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07836-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07860-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07862-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07865-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07865-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07865-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07866-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07867-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07867-study2-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07869-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07870-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07879-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07885-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07896-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07896-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07907-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07908-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07917-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07934-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07935-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07939-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07944-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07945-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07945-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07953-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07953-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07954-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07955-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07958-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07963-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07967-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07969-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07969-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07972-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07982-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07982-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07983-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07986-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07989-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07991-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07993-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07995-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07996-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient07997-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08004-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08007-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08018-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08021-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08023-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08034-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08037-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08048-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08063-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08064-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08066-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08084-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08094-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08099-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08105-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08111-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08116-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08117-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08118-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08123-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08127-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08128-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08138-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08144-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08157-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08163-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08165-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08179-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08182-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08185-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08191-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08192-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08193-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08194-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08195-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08197-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08199-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08202-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08202-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08202-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08203-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08204-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08211-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08216-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08221-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08222-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08222-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08228-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08237-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08245-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08262-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08276-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08276-study4-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08294-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08294-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08294-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08297-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08298-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08310-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08335-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08340-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08341-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08343-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08348-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08349-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08350-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08354-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08354-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08357-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08365-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08378-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08378-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08380-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08385-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08387-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08392-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08393-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08409-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08411-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08413-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08417-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08421-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08423-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08437-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08439-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08439-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08443-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08445-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08448-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08448-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08449-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08452-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08458-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08461-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08466-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08467-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08469-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08472-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08473-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08484-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08487-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08488-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08491-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08498-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08501-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08505-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08510-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08510-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08510-study2-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08517-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08522-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08527-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08555-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08563-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08566-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08567-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08567-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08579-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08585-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08598-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08616-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08617-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08626-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08627-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08629-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08636-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08643-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08645-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08649-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08650-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08656-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08660-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08661-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08669-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08671-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08673-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08681-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08685-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08685-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08705-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08716-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08717-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08722-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08725-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08725-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08728-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08730-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08735-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08740-study22-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08741-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08743-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08751-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08752-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08758-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08763-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08771-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08798-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08830-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08830-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08833-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08839-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08849-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08860-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08871-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08874-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08879-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08880-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08884-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08885-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08893-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08903-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08907-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08910-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08910-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08926-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08927-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08931-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08937-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08938-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08951-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08960-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08964-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08965-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08977-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08978-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08979-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08984-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08993-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08994-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08996-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient08999-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09000-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09006-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09014-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09023-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09025-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09035-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09039-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09043-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09046-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09048-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09052-study22-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09052-study26-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09068-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09070-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09074-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09075-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09079-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09091-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09104-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09107-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09116-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09129-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09132-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09135-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09146-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09150-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09152-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/No_Finding/patient09156-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Pleural_Other/
      inflating: data_upload_v2/test/non/Pleural_Other/patient00206-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient00927-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient01832-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient02683-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient03509-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient03559-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient04083-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient04083-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient04470-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient04501-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient04986-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient05094-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient06622-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient06794-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient07277-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient08478-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient08620-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient10182-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient10572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient10814-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient10884-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient11122-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient11465-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient12658-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient14632-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient15827-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient17132-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient18522-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient20572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient22211-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient22340-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient22990-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient25354-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient27680-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient27989-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient28371-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient28890-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient29016-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient29191-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient31904-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient31941-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient32425-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient32768-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient33129-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient33956-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient33968-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient35413-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient36297-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient37200-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient38011-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient38163-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient41227-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient41375-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pleural_Other/patient41862-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Pneumonia/
      inflating: data_upload_v2/test/non/Pneumonia/patient00240-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient00394-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient00702-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient01099-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient01165-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient01369-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient01700-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient01737-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient01868-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient02189-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient02471-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient02743-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient02743-study3-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient02979-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient03027-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient03130-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient03419-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient03440-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient03677-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient03980-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient04385-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient04447-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient04721-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient04768-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient04828-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient04840-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient05088-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient05282-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient05341-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient05491-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient06333-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient06486-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient06593-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient06809-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient06922-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient07409-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient07671-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient08016-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient08196-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient08799-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient08799-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient09514-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient09592-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10198-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10361-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10399-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10456-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10588-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10624-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10657-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10657-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient10803-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient11076-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient11580-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient11845-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient12316-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient12572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient12772-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient12792-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient12803-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient13077-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient13077-study4-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14092-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14121-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14213-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14737-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14755-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14770-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient14770-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient15462-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient15650-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient15709-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient15847-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient16081-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient16438-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient16678-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient16988-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient17182-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient17282-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient17302-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient17447-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient17631-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient18057-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient18085-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient18092-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient18167-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient18189-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient18232-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient19284-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20056-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20073-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20285-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20408-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20486-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20621-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20687-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumonia/patient20783-study1-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Pneumothorax/
      inflating: data_upload_v2/test/non/Pneumothorax/patient00005-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00005-study2-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00048-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00055-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00078-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00078-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00078-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00078-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00132-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00255-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00259-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00282-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00310-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00359-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00369-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00426-study33-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00433-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study23-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study25-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study26-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study27-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study30-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study31-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study32-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00467-study33-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00501-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00502-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00555-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00572-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00687-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00713-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00724-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00842-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00863-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00863-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient00863-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01046-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01069-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01161-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01178-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01188-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01201-study2-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01215-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01217-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01217-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01217-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01217-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01259-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01263-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01340-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01342-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01342-study10-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01342-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01344-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01344-study2-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01386-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01394-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01409-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01409-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01409-study4-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01409-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01516-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01516-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01516-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01516-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01570-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01634-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01661-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01705-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01731-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01799-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01799-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01802-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient01917-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02001-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02011-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02049-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02098-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02205-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02333-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02377-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02427-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02427-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02450-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02526-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02593-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02615-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02615-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02615-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02661-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02661-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02680-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02701-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02701-study5-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02759-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient02861-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03012-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03034-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03122-study14-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03122-study21-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03122-study23-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03122-study41-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03199-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03199-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03273-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03273-study4-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03304-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03304-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03304-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03343-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03354-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03569-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Pneumothorax/patient03569-study6-view1_frontal.jpg  
       creating: data_upload_v2/test/non/Support_Devices/
      inflating: data_upload_v2/test/non/Support_Devices/patient00012-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00012-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00053-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00056-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00064-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00067-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00087-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00093-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00102-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00114-study17-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00117-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00117-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00123-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00138-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00150-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00151-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00170-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00177-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00182-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00204-study15-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00233-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00259-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00282-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00308-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00314-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00326-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00359-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00372-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00372-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00375-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00408-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00412-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00430-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00438-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00446-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00466-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00466-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00467-study11-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00467-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00467-study21-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00469-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00472-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00473-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00478-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00487-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00506-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00519-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00526-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00596-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00597-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00603-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00627-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00627-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00628-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00631-study40-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00683-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00689-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00691-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00691-study1-view2_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00691-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00726-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00764-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00764-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00795-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00815-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00836-study16-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00857-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00927-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00930-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00943-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00959-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00974-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient00995-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01010-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01051-study13-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01107-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01124-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01124-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01156-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01161-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01167-study41-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01181-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01284-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01311-study12-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01317-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01335-study6-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01343-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01343-study9-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01356-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01359-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01359-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01401-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01420-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01425-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01448-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01457-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01489-study20-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01498-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01498-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01498-study4-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01498-study5-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01543-study8-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01599-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01630-study10-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01664-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01689-study3-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01717-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01717-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01728-study2-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01744-study7-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01760-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01777-study1-view1_frontal.jpg  
      inflating: data_upload_v2/test/non/Support_Devices/patient01788-study2-view1_frontal.jpg  
       creating: data_upload_v2/train/
       creating: data_upload_v2/train/covid/
      inflating: data_upload_v2/train/covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg  
      inflating: data_upload_v2/train/covid/16664_1_1.jpg  
      inflating: data_upload_v2/train/covid/16691_1_1.jpg  
      inflating: data_upload_v2/train/covid/171CB377-62FF-4B76-906C-F3787A01CB2E.jpeg  
      inflating: data_upload_v2/train/covid/1B734A89-A1BF-49A8-A1D3-66FAFA4FAC5D.jpeg  
      inflating: data_upload_v2/train/covid/1-s2.0-S1684118220300608-main.pdf-001.jpg  
      inflating: data_upload_v2/train/covid/1-s2.0-S1684118220300608-main.pdf-002.jpg  
      inflating: data_upload_v2/train/covid/1-s2.0-S1684118220300682-main.pdf-002-a1.png  
      inflating: data_upload_v2/train/covid/1-s2.0-S1684118220300682-main.pdf-002-a2.png  
      inflating: data_upload_v2/train/covid/23E99E2E-447C-46E5-8EB2-D35D12473C39.png  
      inflating: data_upload_v2/train/covid/2966893D-5DDF-4B68-9E2B-4979D5956C8E.jpeg  
      inflating: data_upload_v2/train/covid/2B8649B2-00C4-4233-85D5-1CE240CF233B.jpeg  
      inflating: data_upload_v2/train/covid/2C10A413-AABE-4807-8CCE-6A2025594067.jpeg  
      inflating: data_upload_v2/train/covid/2C26F453-AF3B-4517-BB9E-802CF2179543.jpeg  
      inflating: data_upload_v2/train/covid/2-chest-filmc.jpg  
      inflating: data_upload_v2/train/covid/31BA3780-2323-493F-8AED-62081B9C383B.jpeg  
      inflating: data_upload_v2/train/covid/353889E0-A1E8-4F9E-A0B8-F24F36BCFBFB.jpeg  
      inflating: data_upload_v2/train/covid/39EE8E69-5801-48DE-B6E3-BE7D1BCF3092.jpeg  
      inflating: data_upload_v2/train/covid/4e43e48d52c9e2d4c6c1fb9bc1544f_jumbo.jpeg  
      inflating: data_upload_v2/train/covid/4-x-day1.jpg  
      inflating: data_upload_v2/train/covid/4-x-day13.jpg  
      inflating: data_upload_v2/train/covid/4-x-day4.jpg  
      inflating: data_upload_v2/train/covid/4-x-day8.jpg  
      inflating: data_upload_v2/train/covid/5931B64A-7B97-485D-BE60-3F1EA76BC4F0.jpeg  
      inflating: data_upload_v2/train/covid/5A78BCA9-5B7A-440D-8A4E-AE7710EA6EAD.jpeg  
      inflating: data_upload_v2/train/covid/5e6dd879fde9502400e58b2f.jpeg  
      inflating: data_upload_v2/train/covid/67d668e570c242404ba82c7cbe2ca8f2-0015-0.jpg  
      inflating: data_upload_v2/train/covid/67d668e570c242404ba82c7cbe2ca8f2-05be-0.jpg  
      inflating: data_upload_v2/train/covid/6CB4EFC6-68FA-4CD5-940C-BEFA8DAFE9A7.jpeg  
      inflating: data_upload_v2/train/covid/7E335538-2F86-424E-A0AB-6397783A38D0.jpeg  
      inflating: data_upload_v2/train/covid/80446565-E090-4187-A031-9D3CEAA586C8.jpeg  
      inflating: data_upload_v2/train/covid/85E52EB3-56E9-4D67-82DA-DEA247C82886.jpeg  
      inflating: data_upload_v2/train/covid/8FDE8DBA-CFBD-4B4C-B1A4-6F36A93B7E87.jpeg  
      inflating: data_upload_v2/train/covid/93FE0BB1-022D-4F24-9727-987A07975FFB.jpeg  
      inflating: data_upload_v2/train/covid/9C34AF49-E589-44D5-92D3-168B3B04E4A6.jpeg  
      inflating: data_upload_v2/train/covid/ae6c954c0039de4b5edee53865ffee43-e6c8-0.jpg  
      inflating: data_upload_v2/train/covid/B59DD164-51D5-40DF-A926-6A42DD52EBE8.jpeg  
      inflating: data_upload_v2/train/covid/C6EA0BE5-B01E-4113-B194-18D956675E25.jpeg  
      inflating: data_upload_v2/train/covid/covid-19-caso-70-1-PA.jpg  
      inflating: data_upload_v2/train/covid/covid-19-caso-70-2-APS.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-15-PA.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-2.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-22-day1-pa.png  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-22-day2-pa.png  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-34.png  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-35-1.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-35-2.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-40.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-41-day-0.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-41-day-2.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-43-day0.jpeg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-43-day2.jpeg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-7-PA.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-evolution-over-a-week-1-day0-PA.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-evolution-over-a-week-1-day3.jpg  
      inflating: data_upload_v2/train/covid/covid-19-pneumonia-evolution-over-a-week-1-day4.jpg  
      inflating: data_upload_v2/train/covid/E63574A7-4188-4C8D-8D17-9D67A18A1AFA.jpeg  
      inflating: data_upload_v2/train/covid/extubation-1.jpg  
      inflating: data_upload_v2/train/covid/extubation-13.jpg  
      inflating: data_upload_v2/train/covid/extubation-4.jpg  
      inflating: data_upload_v2/train/covid/extubation-8.jpg  
      inflating: data_upload_v2/train/covid/F2DE909F-E19C-4900-92F5-8F435B031AC6.jpeg  
      inflating: data_upload_v2/train/covid/F4341CE7-73C9-45C6-99C8-8567A5484B63.jpeg  
      inflating: data_upload_v2/train/covid/FE9F9A5D-2830-46F9-851B-1FF4534959BE.jpeg  
      inflating: data_upload_v2/train/covid/figure1-5e71be566aa8714a04de3386-98-left.jpeg  
      inflating: data_upload_v2/train/covid/figure1-5e75d0940b71e1b702629659-98-right.jpeg  
      inflating: data_upload_v2/train/covid/figure1-5e7c1b8d98c29ab001275405-98.jpeg  
      inflating: data_upload_v2/train/covid/figure1-5e7c1b8d98c29ab001275405-98-later.jpeg  
      inflating: data_upload_v2/train/covid/gr1_lrg-a.jpg  
      inflating: data_upload_v2/train/covid/kjr-21-e24-g001-l-a.jpg  
      inflating: data_upload_v2/train/covid/kjr-21-e24-g002-l-a.jpg  
      inflating: data_upload_v2/train/covid/lancet-case2a.jpg  
      inflating: data_upload_v2/train/covid/lancet-case2b.jpg  
      inflating: data_upload_v2/train/covid/nejmoa2001191_f3-PA.jpeg  
      inflating: data_upload_v2/train/covid/nejmoa2001191_f4.jpeg  
      inflating: data_upload_v2/train/covid/nejmoa2001191_f5-PA.jpeg  
      inflating: data_upload_v2/train/covid/paving.jpg  
      inflating: data_upload_v2/train/covid/radiol.2020201160.fig2b.jpeg  
      inflating: data_upload_v2/train/covid/radiol.2020201160.fig2d.jpeg  
      inflating: data_upload_v2/train/covid/radiol.2020201160.fig3a.jpeg  
      inflating: data_upload_v2/train/covid/radiol.2020201160.fig3c.jpeg  
      inflating: data_upload_v2/train/covid/radiol.2020201160.fig3d.jpeg  
      inflating: data_upload_v2/train/covid/RX-torace-a-letto-del-paziente-in-unica-proiezione-AP-1-1.jpeg  
      inflating: data_upload_v2/train/covid/ryct.2020200034.fig2.jpeg  
       creating: data_upload_v2/train/non/
      inflating: data_upload_v2/train/non/Atelectasis-patient04061-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient05424-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient06524-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient08675-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient09344-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient17372-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient18918-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient19049-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient20036-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient20272-study17-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient20318-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient25837-study35-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient26569-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient28865-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient30594-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient35833-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient37272-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient44217-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient44351-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient48245-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient50154-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient53621-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient56228-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient60559-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient61402-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Atelectasis-patient63795-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient01683-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient03124-study11-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient03541-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient07942-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient14429-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient19896-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient22684-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient27368-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient27490-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient30449-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient31090-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient33139-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient34065-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient39872-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient39992-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient41014-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient49614-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient53651-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient56407-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient56816-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient57609-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient59246-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient62259-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Cardiomegaly-patient62346-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient06484-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient13828-study9-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient14143-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient15874-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient16216-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient17196-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient18602-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient25095-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient26518-study40-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient27286-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient28761-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient31053-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient31120-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient33854-study13-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient34616-study14-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient34934-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient35064-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient35241-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient35295-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient36884-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient38079-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient39118-study8-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient40036-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient40886-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient40886-study6-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient41594-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient42423-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient44532-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient48904-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient49048-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient49558-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient50281-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Consolidation-patient61878-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient00694-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient02458-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient09672-study2-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient12228-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient17197-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient17698-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient18283-study9-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient19356-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient26327-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient27176-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient28855-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient29014-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient29015-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient29766-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient31848-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient32972-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient33854-study15-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient36506-study14-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient36975-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient37639-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient37688-study12-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient37688-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient38949-study13-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient39837-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient39866-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient41108-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient42393-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient46659-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient47954-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient48290-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient48964-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient50216-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient52488-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient59820-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Edema-patient63015-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient01660-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient03568-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient04057-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient06247-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient08253-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient14428-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient17498-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient17905-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient18131-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient20846-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient24585-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient24983-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient25250-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient30350-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient31917-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient33943-study9-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient34848-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient37621-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient37898-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient38952-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient47018-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient51161-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient52072-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient52877-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient54362-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient56317-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient60561-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Enlarged Cardiomediastinum-patient64300-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient00058-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient00158-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient00208-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient02805-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient03162-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient04619-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient06368-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient07145-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient09610-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient10460-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient12226-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient14492-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient21610-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient23671-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient24601-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient30339-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient31060-study14-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient31760-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient33608-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient44443-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient47834-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient49928-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Fracture-patient54859-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient06109-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient09669-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient09993-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient14762-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient15441-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient18645-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient22959-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient25683-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient28657-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient30668-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient32369-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient33562-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient37521-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient44761-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient47992-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient55655-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient56124-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Lesion-patient56401-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient01141-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient02143-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient03633-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient04034-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient04707-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient07080-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient08024-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient09109-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient12608-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient13115-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient15333-study11-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient16019-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient17382-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient18878-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient24570-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient24820-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient26200-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient26489-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient26562-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient34268-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient34512-study12-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient34993-study9-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient40028-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient40268-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient43497-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient44245-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient45892-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient49715-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient52551-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient52960-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient53592-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient56630-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient58232-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient59050-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient59816-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient60973-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Lung Opacity-patient64055-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient01127-study14-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient01429-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient01572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient01617-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient02123-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03180-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03278-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03306-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03659-study17-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03675-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03837-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient03906-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient04064-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient04692-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06257-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06311-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06403-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06458-study25-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06527-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06547-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06869-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06907-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient06946-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient07089-study2-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient07433-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient08176-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient08902-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient09010-study11-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient09162-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient10418-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient10505-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient10633-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient10789-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient10963-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient11010-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient11266-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient11513-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient11642-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient11881-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient12295-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient12754-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient13737-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient13797-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient13824-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient14380-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient15439-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient15540-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient15761-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient16105-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient16159-study11-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient16184-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient16358-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient16629-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient16904-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient17231-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient17301-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient17712-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient18271-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient18315-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient18357-study19-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient18667-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient19602-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient19628-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient19725-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient20243-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient20323-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient20328-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient20339-study12-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient20843-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient20939-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient21164-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient21399-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient21654-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient21786-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient22116-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient22594-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient23026-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient23220-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient23329-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient23517-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient23856-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient24195-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient24219-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient24422-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient24901-study8-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient25064-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient25375-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient25421-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient25729-study16-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient26018-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient26450-study29-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient26541-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient26763-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient27209-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient27523-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient27963-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient27977-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient28824-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29015-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29323-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29355-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29363-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29363-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29719-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29851-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient29950-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient30124-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient30324-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient30518-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient30781-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient30931-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient31160-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient31965-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient33132-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient33573-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient34325-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient34586-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient34909-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient35000-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient35193-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient35430-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient35695-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient36171-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient36302-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient36436-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient36465-study14-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient37597-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient37874-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient37889-study11-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient38779-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient38879-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient38955-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient39317-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient39364-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient39420-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient39451-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient40010-study8-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient40452-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient40760-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient40899-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient41564-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient41847-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient41866-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient42056-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient44218-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient44906-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient45080-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient45277-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient46068-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient46142-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient46388-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient46458-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient46679-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient46893-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient47707-study1-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient48065-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient48258-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient48471-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient49302-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient49558-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient49936-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient50021-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient50050-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient50141-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient50301-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient51324-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient51660-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient51764-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient52153-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient52236-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient52624-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient53227-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient53508-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient53712-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient53797-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient54222-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient54409-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient55177-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient55847-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient56387-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient56406-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient57341-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient57448-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient57481-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient58001-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient58385-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient58688-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient58740-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient59807-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient59984-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient61035-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient61343-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient61525-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient61659-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient61693-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient61743-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient63560-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/No Finding-patient63904-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient00414-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient01043-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient03705-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient05843-study11-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient06730-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient09530-study39-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient17572-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient17869-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient18674-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient20835-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient25572-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient30617-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient30718-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient30972-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient35099-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient35290-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient36661-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient37517-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient40287-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient40304-study17-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient44368-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient50733-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient52567-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient59226-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient59395-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Effusion-patient60775-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient00855-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient02866-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient05328-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient10887-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient17818-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient19843-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient27293-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient28364-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient30617-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient32290-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient33734-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient34920-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient34954-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient40806-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient45499-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient50407-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient50506-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient57120-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pleural Other-patient59134-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient01043-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient05831-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient06552-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient10081-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient11586-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient14645-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient16755-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient24522-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient26337-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient26582-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient26588-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient27861-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient28516-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient30349-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient48660-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient49120-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient54176-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient54711-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient57909-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient58142-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumonia-patient61672-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient02703-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient06761-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient07030-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient08173-study28-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient08173-study30-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient10368-study13-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient10756-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient10895-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient11890-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient12791-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient14395-study8-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient14771-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient15259-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient15314-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient16635-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient16810-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient17529-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient18048-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient19395-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient19968-study20-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient20036-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient20442-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient21088-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient25588-study8-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient26025-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient26134-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient27367-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient28015-study28-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient30731-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient31120-study30-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient34807-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient34974-study2-view2_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient35615-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient36093-study23-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient37042-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient37625-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient39665-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient40543-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient40677-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient41127-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient41174-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient42027-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient42027-study7-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient42847-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient44356-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient45952-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient47445-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient51207-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient51503-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient51990-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Pneumothorax-patient54371-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient00283-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient02339-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient03132-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient05991-study10-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient06709-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient14451-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient15136-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient17193-study18-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient20998-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient21187-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient22213-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient22954-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient23508-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient23730-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient26111-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient26949-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient27543-study47-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient27843-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient30011-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient31283-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient31828-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient33552-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient35783-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient36884-study8-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient38008-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient38943-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient39445-study13-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient39527-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient40597-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient41425-study4-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient43372-study5-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient43476-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient47030-study3-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient47189-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient48654-study6-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient50424-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient53734-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient53737-study2-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient54504-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient57841-study1-view1_frontal.jpg  
      inflating: data_upload_v2/train/non/Support Devices-patient61366-study2-view1_frontal.jpg  



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

Para o cenário apresentado, podem haver casos em que tenhamos uma probabilidade de classificação baixa, tornando necessário uma validação de um médico especialista. 

A validação humana é um caso comum para workloads de machine learning em que o modelo tenha uma resposta com probabilidade abaixo de um determinado threshold. Para solucionar esse problema a AWS dispõe do serviço Augmented IA, que será o assunto da parte 2 desse blogpost.


```python

```
