from typing import List, Any
from pydantic import BaseModel, Field

class ProcessDocumentViewModel(BaseModel):
    text: str = Field(
        description="Teksts kurš tiks apstrādāts",
        min_length=1,
        examples=[
            "Atbildētājs Andis Bērziņš rakstveida paskaidrojumus šī gada 11. Aprīli nav iesniedzis. Paskaidrojuma veidlapa nosūtīta uz Bērziņa deklarēto adresi Ventspilī, “Urgas” – 5/56, Pampāļi, LV-2323, kadastra numurs 80042323455555. Papildus nosūtīta arī uz norādīto dzīves vietu Atmodas iela 55 K3 – 38A, Ugāle, LV-2525. Papildus ir noskaidrots, ka Atbildētājs nav laicīgi iegādājies OCTA savam transporta līdzeklim Mercedes Benz GT67, reģ.nr. AL-3401. Nauda ir ieskaitāma atbildētāja kontā LV76HABA0551041569800. Atbildētājs Andris Bērziņš uz jauno kontu LV76HABA0551041569899 ir pārskaitījis daļu no naudas līdzekļiem. Taču kontam Nr. LV76HABA0551041569800 ir cita banka. Papildus noskaidrots, ka Andris Bērziņš ir laulībā ar Andu Bērziņu - Kalniņu (pirmslaulību uzvārds Kalniņa), p.k. 121281-99999. Anda Bērziņa- Kalniņa lūdz tiesu piedzīt no atbildētāja prasītājas labā uzturlīdzekļus bērnu Almas Bērziņas - Kalniņas un Artūra Bērziņa- Kalniņa uzturam 30% apmērā. Alma un Artūrs dzīvo pie prasītājas. Uzturlīdzekļus lūdz ieskaitīt kontā Nr. LV76HABA0551041569800 vai otrā kontā LV76HABA0551041569844. Prasītājai pieder nekustamais īpašums “Palācīši”, Siguldā, kadastra numurs 0110033445556. Un automašīna Volvo 6, reģistrācijas numurs KA5567. Atbildētājs zvanījis Prasītājai no telefona nr.29292929. Tiesnese Andra Bērziņa izskatīja lietu 2024. Gada 15. Maijā."
        ],
    )

class ProcessDocumentResponseViewModel(BaseModel):
    text: str = Field(
        description="Teksts kurš tika apstrādāts"
    )
    elapsedSeconds: float
    wordsPerSecond: float
    usedDevice: str
    level1Entities: List[Any]
    level2Entities: List[Any]
