import re
import json

# Dữ liệu HTML gốc
html = """
<a href="/vie/thuoc/abobotulinum-toxin-a-8" class="name_drug">Abobotulinum Toxin A</a>
<a href="/vie/thuoc/abobotulinum-toxin-a-108" class="name_drug">Abobotulinum Toxin A</a>
<a href="/vie/thuoc/acarbose-109" class="name_drug">Acarbose</a>
<a href="/vie/thuoc/acenocoumarol-110" class="name_drug">Acenocoumarol</a>
<a href="/vie/thuoc/acetazolamide-111" class="name_drug">Acetazolamide</a>
<a href="/vie/thuoc/acetyl-dl-leucine-88" class="name_drug">Acetyl-Dl-Leucine</a>
<a href="/vie/thuoc/acetylcysteine-112" class="name_drug">Acetylcysteine</a>
<a href="/vie/thuoc/acid-folic-113" class="name_drug">Acid Folic</a>
<a href="/vie/thuoc/acid-thioctic-114" class="name_drug">Acid Thioctic</a>
<a href="/vie/thuoc/acid-ursodeoxycholic-115" class="name_drug">Acid Ursodeoxycholic</a>
<a href="/vie/thuoc/acyclovir-tra-mat-116" class="name_drug">Acyclovir (Tra mắt)</a>
<a href="/vie/thuoc/acyclovir-uong-117" class="name_drug">Acyclovir (Uống)</a>
<a href="/vie/thuoc/adalimumab-118" class="name_drug">Adalimumab</a>
<a href="/vie/thuoc/adapalene-119" class="name_drug">Adapalene</a>
<a href="/vie/thuoc/adapalene-clindamycin-120" class="name_drug">Adapalene/ Clindamycin</a>
<a href="/vie/thuoc/adenosine-121" class="name_drug">Adenosine</a>
<a href="/vie/thuoc/adrenaline-122" class="name_drug">Adrenaline</a>
<a href="/vie/thuoc/afatinib-123" class="name_drug">Afatinib</a>
<a href="/vie/thuoc/albendazole-124" class="name_drug">Albendazole</a>
<a href="/vie/thuoc/albumin-125" class="name_drug">Albumin</a>
<a href="/vie/thuoc/alendronate-75" class="name_drug">Alendronate</a>
<a href="/vie/thuoc/alendronate-126" class="name_drug">Alendronate</a>
<a href="/vie/thuoc/alfuzosin-19" class="name_drug">Alfuzosin</a>
<a href="/vie/thuoc/alimemazine-127" class="name_drug">Alimemazine</a>
<a href="/vie/thuoc/bambuterol-31" class="name_drug">Bambuterol</a>
<a href="/vie/thuoc/bernevit-47" class="name_drug">Bernevit</a>
<a href="/vie/thuoc/betahistin-36" class="name_drug">Betahistin</a>
<a href="/vie/thuoc/bevacizumab-23" class="name_drug">Bevacizumab</a>
<a href="/vie/thuoc/bisoprolol-38" class="name_drug">Bisoprolol</a>
<a href="/vie/thuoc/broncho-vacxom-43" class="name_drug">Broncho Vacxom</a>
<a href="/vie/thuoc/budesonide-khi-dung-37" class="name_drug">Budesonide (Khí dung)</a>
<a href="/vie/thuoc/capecitabine-39" class="name_drug">Capecitabine</a>
<a href="/vie/thuoc/carvedilol-52" class="name_drug">Carvedilol</a>
<a href="/vie/thuoc/caspofungin-40" class="name_drug">Caspofungin</a>
<a href="/vie/thuoc/ceftriaxone-13" class="name_drug">Ceftriaxone</a>
<a href="/vie/thuoc/celecoxib-44" class="name_drug">Celecoxib</a>
<a href="/vie/thuoc/cerebrolysin-48" class="name_drug">Cerebrolysin</a>
<a href="/vie/thuoc/cilostazol-16" class="name_drug">Cilostazol</a>
<a href="/vie/thuoc/clopidogrel-42" class="name_drug">Clopidogrel</a>
<a href="/vie/thuoc/clotrimazole-55" class="name_drug">Clotrimazole</a>
<a href="/vie/thuoc/colchicine-53" class="name_drug">Colchicine</a>
<a href="/vie/thuoc/dabigatran-54" class="name_drug">Dabigatran</a>
<a href="/vie/thuoc/dapagliflozin-107" class="name_drug">Dapagliflozin</a>
<a href="/vie/thuoc/desloratadine-60" class="name_drug">Desloratadine</a>
<a href="/vie/thuoc/diacerein-32" class="name_drug">Diacerein</a>
<a href="/vie/thuoc/diclofenac-tiem-46" class="name_drug">Diclofenac (Tiêm)</a>
<a href="/vie/thuoc/diphenhydramine-58" class="name_drug">Diphenhydramine</a>
<a href="/vie/thuoc/enoxaparin-62" class="name_drug">Enoxaparin</a>
<a href="/vie/thuoc/entecavir-66" class="name_drug">Entecavir</a>
<a href="/vie/thuoc/ertapenem-67" class="name_drug">Ertapenem</a>
<a href="/vie/thuoc/esomeprazole-50" class="name_drug">Esomeprazole</a>
<a href="/vie/thuoc/etoricoxib-25" class="name_drug">Etoricoxib</a>
<a href="/vie/thuoc/ezetimibe-63" class="name_drug">Ezetimibe</a>
<a href="/vie/thuoc/febuxostat-64" class="name_drug">Febuxostat</a>
<a href="/vie/thuoc/felodipine-65" class="name_drug">Felodipine</a>
<a href="/vie/thuoc/fenofibrate-29" class="name_drug">Fenofibrate</a>
<a href="/vie/thuoc/fentanyl-mieng-dan-61" class="name_drug">Fentanyl (Miếng dán)</a>
<a href="/vie/thuoc/fexofenadin-68" class="name_drug">Fexofenadin</a>
<a href="/vie/thuoc/fleet-enema-57" class="name_drug">Fleet Enema</a>
<a href="/vie/thuoc/fluconazole-56" class="name_drug">Fluconazole</a>
<a href="/vie/thuoc/fluconazole-6" class="name_drug">Fluconazole</a>
<a href="/vie/thuoc/gabapentin-103" class="name_drug">Gabapentin</a>
<a href="/vie/thuoc/ganciclovir-101" class="name_drug">Ganciclovir</a>
<a href="/vie/thuoc/gaviscon-102" class="name_drug">Gaviscon</a>
<a href="/vie/thuoc/gliclazide-105" class="name_drug">Gliclazide</a>
<a href="/vie/thuoc/glyceryl-trinitrate-uong-77" class="name_drug">Glyceryl Trinitrate (Uống)</a>
<a href="/vie/thuoc/goserelin-71" class="name_drug">Goserelin</a>
<a href="/vie/thuoc/gabapentin-103" class="name_drug">Gabapentin</a>
<a href="/vie/thuoc/ganciclovir-101" class="name_drug">Ganciclovir</a>
<a href="/vie/thuoc/gaviscon-102" class="name_drug">Gaviscon</a>
<a href="/vie/thuoc/gliclazide-105" class="name_drug">Gliclazide</a>
<a href="/vie/thuoc/glyceryl-trinitrate-uong-77" class="name_drug">Glyceryl Trinitrate (Uống)</a>
<a href="/vie/thuoc/goserelin-71" class="name_drug">Goserelin</a>
<a href="/vie/thuoc/haloperidol-72" class="name_drug">Haloperidol</a>
<a href="/vie/thuoc/hydroxychloroquine-74" class="name_drug">Hydroxychloroquine</a>
<a href="/vie/thuoc/hyoscin-n-butylbromide-14" class="name_drug">Hyoscin-N-Butylbromide</a>
<a href="/vie/thuoc/ibuprofen-99" class="name_drug">Ibuprofen</a>
<a href="/vie/thuoc/indapamide-91" class="name_drug">Indapamide</a>
<a href="/vie/thuoc/irbesartan-26" class="name_drug">Irbesartan</a>
<a href="/vie/thuoc/ivabradine-97" class="name_drug">Ivabradine</a>
<a href="/vie/thuoc/ibuprofen-99" class="name_drug">Ibuprofen</a>
<a href="/vie/thuoc/indapamide-91" class="name_drug">Indapamide</a>
<a href="/vie/thuoc/irbesartan-26" class="name_drug">Irbesartan</a>
<a href="/vie/thuoc/ivabradine-97" class="name_drug">Ivabradine</a>
<a href="/vie/thuoc/ibuprofen-99" class="name_drug">Ibuprofen</a>
<a href="/vie/thuoc/indapamide-91" class="name_drug">Indapamide</a>
<a href="/vie/thuoc/irbesartan-26" class="name_drug">Irbesartan</a>
<a href="/vie/thuoc/ivabradine-97" class="name_drug">Ivabradine</a>
<a href="/vie/thuoc/lactulose-82" class="name_drug">Lactulose</a>
<a href="/vie/thuoc/levetiracetam-83" class="name_drug">Levetiracetam</a>
<a href="/vie/thuoc/levodopabenserazide-84" class="name_drug">Levodopa/Benserazide</a>
<a href="/vie/thuoc/levothyroxine-35" class="name_drug">Levothyroxine</a>
<a href="/vie/thuoc/linezolid-93" class="name_drug">Linezolid</a>
<a href="/vie/thuoc/lipofundin-mctlct-86" class="name_drug">Lipofundin MCT/LCT</a>
<a href="/vie/thuoc/lisinopril-87" class="name_drug">Lisinopril</a>
<a href="/vie/thuoc/loperamide-92" class="name_drug">Loperamide</a>
<a href="/vie/thuoc/loratadine-59" class="name_drug">Loratadine</a>
<a href="/vie/thuoc/losartan-90" class="name_drug">Losartan</a>
<a href="/vie/thuoc/lactulose-82" class="name_drug">Lactulose</a>
<a href="/vie/thuoc/levetiracetam-83" class="name_drug">Levetiracetam</a>
<a href="/vie/thuoc/levodopabenserazide-84" class="name_drug">Levodopa/Benserazide</a>
<a href="/vie/thuoc/levothyroxine-35" class="name_drug">Levothyroxine</a>
<a href="/vie/thuoc/linezolid-93" class="name_drug">Linezolid</a>
<a href="/vie/thuoc/lipofundin-mctlct-86" class="name_drug">Lipofundin MCT/LCT</a>
<a href="/vie/thuoc/lisinopril-87" class="name_drug">Lisinopril</a>
<a href="/vie/thuoc/loperamide-92" class="name_drug">Loperamide</a>
<a href="/vie/thuoc/loratadine-59" class="name_drug">Loratadine</a>
<a href="/vie/thuoc/losartan-90" class="name_drug">Losartan</a>
<a href="/vie/thuoc/magnesi-lactate-vitamin-b6-197" class="name_drug">Magnesi Lactate/ Vitamin B6</a>
<a href="/vie/thuoc/magnesi-sulfate-196" class="name_drug">Magnesi Sulfate</a>
<a href="/vie/thuoc/magnesi-sulfate-saccharomyces-cerevisiae-195" class="name_drug">Magnesi Sulfate/ Saccharomyces Cerevisiae</a>
<a href="/vie/thuoc/mannitol-194" class="name_drug">Mannitol</a>
<a href="/vie/thuoc/maxitrol-193" class="name_drug">Maxitrol</a>
<a href="/vie/thuoc/mebendazole-192" class="name_drug">Mebendazole</a>
<a href="/vie/thuoc/mebeverine-191" class="name_drug">Mebeverine</a>
<a href="/vie/thuoc/meloxicam-45" class="name_drug">Meloxicam</a>
<a href="/vie/thuoc/menotrophin-190" class="name_drug">Menotrophin</a>
<a href="/vie/thuoc/mephenesin-189" class="name_drug">Mephenesin</a>
<a href="/vie/thuoc/mepivacaine-188" class="name_drug">Mepivacaine</a>
<a href="/vie/thuoc/mercaptopurine-187" class="name_drug">Mercaptopurine</a>
<a href="/vie/thuoc/meropenem-186" class="name_drug">Meropenem</a>
<a href="/vie/thuoc/mesalazine-95" class="name_drug">Mesalazine</a>
<a href="/vie/thuoc/mesna-185" class="name_drug">Mesna</a>
<a href="/vie/thuoc/metformin-106" class="name_drug">Metformin</a>
<a href="/vie/thuoc/methotrexate-184" class="name_drug">Methotrexate</a>
<a href="/vie/thuoc/methycobalamin-183" class="name_drug">Methycobalamin</a>
<a href="/vie/thuoc/methyldopa-96" class="name_drug">Methyldopa</a>
<a href="/vie/thuoc/methylergometrine-maleate-182" class="name_drug">Methylergometrine Maleate</a>
<a href="/vie/thuoc/methylprednisolone-181" class="name_drug">Methylprednisolone</a>
<a href="/vie/thuoc/metoclopramide-180" class="name_drug">Metoclopramide</a>
<a href="/vie/thuoc/metoprolol-34" class="name_drug">Metoprolol</a>
<a href="/vie/thuoc/metronidazole-tiem-179" class="name_drug">Metronidazole (Tiêm)</a>
<a href="/vie/thuoc/naproxen-160" class="name_drug">Naproxen</a>
<a href="/vie/thuoc/natri-chloride-045-159" class="name_drug">Natri chloride 0,45%</a>
<a href="/vie/thuoc/natri-chloride-10-158" class="name_drug">Natri chloride 10%</a>
<a href="/vie/thuoc/natri-hyaluronate-157" class="name_drug">Natri Hyaluronate</a>
<a href="/vie/thuoc/natri-hypochloride-156" class="name_drug">Natri hypochloride</a>
<a href="/vie/thuoc/natri-polystyrene-sulfonate-155" class="name_drug">Natri polystyrene sulfonate</a>
<a href="/vie/thuoc/natri-sulphacetamide-154" class="name_drug">Natri sulphacetamide</a>
<a href="/vie/thuoc/neoamyu-153" class="name_drug">Neoamyu</a>
<a href="/vie/thuoc/neopeptin-151" class="name_drug">Neopeptin</a>
<a href="/vie/thuoc/neopeptin-syrup-152" class="name_drug">Neopeptin syrup</a>
<a href="/vie/thuoc/nepafenac-150" class="name_drug">Nepafenac</a>
<a href="/vie/thuoc/nepidermin-rhegf-149" class="name_drug">Nepidermin (rhEGF)</a>
<a href="/vie/thuoc/neuroaid-148" class="name_drug">Neuroaid</a>
<a href="/vie/thuoc/ondansetron-79" class="name_drug">Ondansetron</a>
<a href="/vie/thuoc/oseltamivir-80" class="name_drug">Oseltamivir</a>
<a href="/vie/thuoc/oxycodone-78" class="name_drug">Oxycodone</a>
<a href="/vie/thuoc/pantoprazole-49" class="name_drug">Pantoprazole</a>
<a href="/vie/thuoc/paracetamol-1" class="name_drug">Paracetamol</a>
<a href="/vie/thuoc/perindorpil-76" class="name_drug">Perindorpil</a>
<a href="/vie/thuoc/piracetam-tiem-15" class="name_drug">Piracetam (Tiêm)</a>
<a href="/vie/thuoc/pregabalin-104" class="name_drug">Pregabalin</a>
<a href="/vie/thuoc/protamine-70" class="name_drug">Protamine</a>
<a href="/vie/thuoc/rabeprazole-51" class="name_drug">Rabeprazole</a>
<a href="/vie/thuoc/racecadotril-18" class="name_drug">Racecadotril</a>
<a href="/vie/thuoc/rivaroxaban-69" class="name_drug">Rivaroxaban</a>
<a href="/vie/thuoc/rosuvastatin-28" class="name_drug">Rosuvastatin</a>
<a href="/vie/thuoc/sildenafil-73" class="name_drug">Sildenafil</a>
<a href="/vie/thuoc/sitagliptin-21" class="name_drug">Sitagliptin</a>
<a href="/vie/thuoc/sildenafil-73" class="name_drug">Sildenafil</a>
<a href="/vie/thuoc/sitagliptin-21" class="name_drug">Sitagliptin</a>
<a href="/vie/thuoc/ticagrelor-41" class="name_drug">Ticagrelor</a>
<a href="/vie/thuoc/tiotropium-17" class="name_drug">Tiotropium</a>
<a href="/vie/thuoc/valsartan-89" class="name_drug">Valsartan</a>
<a href="/vie/thuoc/vitamin-k1-phytonadione-147" class="name_drug">Vitamin K1 (Phytonadione)</a>
<a href="/vie/thuoc/vitamin-k2-menatetrenone-146" class="name_drug">Vitamin K2 (Menatetrenone)</a>
<a href="/vie/thuoc/warfarin-145" class="name_drug">Warfarin</a>
<a href="/vie/thuoc/xanh-methylen-143" class="name_drug">Xanh methylen</a>
<a href="/vie/thuoc/xylometazoline-142" class="name_drug">Xylometazoline</a>
<a href="/vie/thuoc/zoledronic-acid-20" class="name_drug">Zoledronic Acid</a>
<a href="/vie/thuoc/zolpidem-141" class="name_drug">Zolpidem</a>
<a href="/vie/thuoc/zopiclon-144" class="name_drug">Zopiclon</a>
<a href="/vie/thuoc/zopiclon-140" class="name_drug">Zopiclon</a>
<a href="/vie/thuoc/allopurinol-9" class="name_drug">Allopurinol</a>
<a href="/vie/thuoc/alpha-chymotrypsine-tiem-128" class="name_drug">Alpha Chymotrypsine (Tiêm)</a>
<a href="/vie/thuoc/alpha-chymotrypsine-uong-129" class="name_drug">Alpha Chymotrypsine (Uống)</a>
<a href="/vie/thuoc/alprostadil-130" class="name_drug">Alprostadil</a>
<a href="/vie/thuoc/alteplase-132" class="name_drug">Alteplase</a>
<a href="/vie/thuoc/alverine-131" class="name_drug">Alverine</a>
<a href="/vie/thuoc/alverine-simethicone-133" class="name_drug">Alverine/ Simethicone</a>
<a href="/vie/thuoc/alvesin-134" class="name_drug">Alvesin</a>
<a href="/vie/thuoc/ambroxol-135" class="name_drug">Ambroxol</a>
<a href="/vie/thuoc/amikacin-136" class="name_drug">Amikacin</a>
<a href="/vie/thuoc/aminophyllin-137" class="name_drug">Aminophyllin</a>
<a href="/vie/thuoc/aminosteril-n-hepa-8-138" class="name_drug">Aminosteril N-Hepa 8%</a>
<a href="/vie/thuoc/amiodarone-139" class="name_drug">Amiodarone</a>
<a href="/vie/thuoc/amlodipine-11" class="name_drug">Amlodipine</a>
<a href="/vie/thuoc/amoxicillin-clavulanic-acid-tiem-12" class="name_drug">Amoxicillin/ Clavulanic Acid (tiêm)</a>
<a href="/vie/thuoc/amphotericin-b-lipid-complex-10" class="name_drug">Amphotericin B lipid complex</a>
<a href="/vie/thuoc/anastrozole-22" class="name_drug">Anastrozole</a>
<a href="/vie/thuoc/atorvastatin-27" class="name_drug">Atorvastatin</a>
<a href="/vie/thuoc/atosiban-33" class="name_drug">Atosiban</a>
<a href="/vie/thuoc/azithromycin-uong-24" class="name_drug">Azithromycin (Uống)</a>
"""

# Dùng regex để trích xuất tất cả các link trong href
links = re.findall(r'href="([^"]+)"', html)

# Ghi vào file JSON
with open('e:/university/TLCN/ChatBot/data/json/vinmec_medicine.json', 'w', encoding='utf-8') as f:
    json.dump(links, f, ensure_ascii=False, indent=4)