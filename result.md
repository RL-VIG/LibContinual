
# 1. LWF
链接：https://arxiv.org/abs/1606.09282
method| 设定 | Buffer | 复现精度 | 论文精度
------ | -----| ---- | ----- | ------
LwF | (10, 10) | - | 43.00% | 43.56%


# 2. LUCIR 
链接：https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- |----- | ------
LUCIR  | (50, 50) | 2000 | 61.50% | 62.42%
LUCIR   | (50, 25)| 2000 | 56.50% | 57.49%

# 3. OCM 
链接：https://proceedings.mlr.press/v162/guo22g.html
method| 设定 | Buffer |复现精度 | 论文精度
------ | ----- | ----| ----- | ------
OCM  |(10，10)| 1000 | 29.5% | 28.1%
OCM  |(10，10)| 2000 | 36.2% | 35.0%
OCM  |(10，10)| 5000 | 43.3% | 42.4%

# 4. iCaRL  
链接： https://arxiv.org/abs/1611.07725
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- | ----- | ------
iCaRL | (20, 20) | 2000| 48.8% | ~53%

# 5. bic 
链接：https://arxiv.org/abs/1905.13260
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- | ----- | ------
bic | (20, 20) | 2000| - | -

# 6. WA 
链接：https://arxiv.org/abs/1911.07053
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- | ----- | ------
WA | (20, 20) | 2000| - | -

# 7. DER 
链接：https://arxiv.org/abs/2004.07211
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- | ----- | ------
DER | (20, 20) | - | - | -

# 8. EWC
链接： https://arxiv.org/abs/1612.00796
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- | ----- | ------
EWC | (50, 50) | - | 57.3% | 58.5%

# 9. DER (TODO)
链接：https://arxiv.org/pdf/2103.16788.pdf

# 10. ER-AML(TODO)
链接：https://arxiv.org/pdf/2104.05025.pdf

# 11. TAMiL (TODO)
链接：https://openreview.net/pdf?id=-M0TNnyWFT5
method| 设定 | Buffer | 复现精度 | 论文精度
------ | ----- | ----- | ----- | ------
TAMiL | (20, 20) | 500 | 49.3% | 50.11%
TAMiL | (20, 20) | 200 | 40.7% | 41.43% 



-----
method| kind |buffer | acc
--- | --- | --- | ---
LwF  | regularize | - | 43.56
WA | regularize | 2000 | 48.20%
EWC | regularize | - | 57.3%
LUCIR | replay | 2000 | 57.49
OCM | replay | 2000 | 35.03%
iCaRL | replay | 2000 | 53.00%
BIC | replay | 2000 | 57.80%
TAMIL | arch | 2000 | 59.21%
DER | arch | 2000 | 


