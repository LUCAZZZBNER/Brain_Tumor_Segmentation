param([Parameter(Mandatory=$true)][string]$Path)
Add-Type -AssemblyName System.IO.Compression.FileSystem;Add-Type -AssemblyName System.IO.Compression
$p=(Resolve-Path -LiteralPath $Path).Path;$z=[IO.Compression.ZipFile]::Open($p,[IO.Compression.ZipArchiveMode]::Update)
try{$e=$z.GetEntry('word/document.xml');$r=[IO.StreamReader]::new($e.Open());try{$raw=$r.ReadToEnd()}finally{$r.Dispose()};$x=[Xml.XmlDocument]::new();$x.PreserveWhitespace=$true;$x.LoadXml($raw);$u='http://schemas.openxmlformats.org/wordprocessingml/2006/main';$n=[Xml.XmlNamespaceManager]::new($x.NameTable);$n.AddNamespace('w',$u)
function T($q){(($q.SelectNodes('.//w:t',$n)|% InnerText)-join '')};function SetT($q,$v){foreach($c in @($q.ChildNodes)){if(-not($c.LocalName-eq'pPr'-and$c.NamespaceURI-eq$u)){[void]$q.RemoveChild($c)}};$rr=$x.CreateElement('w','r',$u);$tt=$x.CreateElement('w','t',$u);$tt.InnerText=$v;[void]$rr.AppendChild($tt);[void]$q.AppendChild($rr)}
$map=[ordered]@{
'论文中文草稿框架'='论文中文稿';
'摘要（初稿）'='摘要';
'1.4 本文工作与贡献（按当前证据表述）'='1.4 本文工作与贡献';
'4.5 模型对比结果（当前已有数据）'='4.5 模型对比结果';
'4.6 组件消融结果（当前已有数据）'='4.6 组件消融结果';
'6.1 结论（当前证据版）'='6.1 结论';
'成稿前重新核对了 8 种配置、3 个训练随机种子的 24 份 test_metrics.json。所有记录的 manifest_sha256 均为 6aee075d9ea8b4a41cdefc372e381a55ec3a099acfaf16002dbde2cd7d5051e7，与 splits/kaggle_3m_multimodal_only_seed42.meta.json 一致；每次测试均包含 525 张切片，其中阳性切片 173 张、空切片 352 张。由原始运行记录重新计算所得表4各项均值与样本标准差均与正文一致。'='一致性复核覆盖 8 种配置、3 个训练随机种子的 24 份 test_metrics.json。所有记录的 manifest_sha256 均为 6aee075d9ea8b4a41cdefc372e381a55ec3a099acfaf16002dbde2cd7d5051e7，与 splits/kaggle_3m_multimodal_only_seed42.meta.json 一致；每次测试均包含 525 张切片，其中阳性切片 173 张、空切片 352 张。由原始运行记录重新计算所得表4各项均值与样本标准差均与正文一致。'
}
foreach($q in $x.SelectNodes('//w:body//w:p',$n)){$v=T $q;if($map.Contains($v)){SetT $q $map[$v]}}
$intro=@($x.SelectNodes('//w:body//w:p',$n)|?{(T $_)-like'草稿范围与事实边界*'});if($intro.Count-ne1){throw 'intro mismatch'};SetT $intro[0] '事实边界：本文以 kaggle_3m 清洁患者级清单和三随机种子实验为主线。主要数据来自 splits/kaggle_3m_multimodal_only_seed42.meta.json、最终消融汇总及对应 runs/ 记录。当前清单包含 104 名患者和 3,629 张切片，训练/验证/测试为 85/9/10 名患者、2,619/485/525 张切片。清单中的 tumor_type 均为 LGG，掩膜用于二值病灶分割；因此，本文不将任务表述为肿瘤类型多分类，也不把内部测试结果解释为外部医院验证或临床诊断性能。'
$e.Delete();$ne=$z.CreateEntry('word/document.xml',[IO.Compression.CompressionLevel]::Optimal);$s=$ne.Open();$set=[Xml.XmlWriterSettings]::new();$set.Encoding=[Text.UTF8Encoding]::new($false);$w=[Xml.XmlWriter]::Create($s,$set);try{$x.Save($w)}finally{$w.Dispose();$s.Dispose()}
}finally{$z.Dispose()}
