from ..adversarial.feature_attacks.FDA import FDA
from ..adversarial.feature_attacks.ILA import ILA
from ..adversarial.feature_attacks.FIA import FIA
from ..adversarial.feature_attacks.NAA import NAA
from ..adversarial.feature_attacks.GFLA.GFLA import GFLA
from ..adversarial.feature_attacks.GFLA.GFLA_D import GFLA_D
from ..adversarial.feature_attacks.ILPD import ILPD
from ..adversarial.logit_attacks.BSR import BSR
from ..adversarial.logit_attacks.ANDA import ANDA
from ..adversarial.logit_attacks.LPAA import LPAA
from ..adversarial.logit_attacks.MIG import MIG
from ..adversarial.logit_attacks.ResPA import ResPA
from ..adversarial.logit_attacks.TGR import TGR
from ..adversarial.logit_attacks.GRA import GRA
from ..adversarial.logit_attacks.PGN import PGN
from ..adversarial.feature_attacks.TAP import TAP
from ..adversarial.feature_attacks.GFLA.GFLA_BSR import GFLA_BSR
from ..adversarial.feature_attacks.GFLA.GFLA_SI import GFLA_SI
from ..adversarial.feature_attacks.GFLA.GFLA_DI import GFLA_DI
from torchattacks import BIM, MIFGSM, NIFGSM, DIFGSM, TIFGSM, SINIFGSM, VMIFGSM

from typing import Iterable, List, Type

ATTACKS = {
    cls.__name__: cls
    for cls in (
        FDA, ILA, FIA, NAA, ILPD, TAP,
        GFLA, GFLA_D, GFLA_BSR, GFLA_SI, GFLA_DI,
        BSR, ANDA, LPAA, MIG, ResPA, TGR, GRA, PGN,
        BIM, MIFGSM, NIFGSM, DIFGSM, TIFGSM, SINIFGSM, VMIFGSM,
    )
}


def get_attacks(names: Iterable[str]) -> List[Type]:
    """Retrieves attack classes by name from registry."""
    unknown = [n for n in names if n not in ATTACKS]
    if unknown:
        raise ValueError(f"Unknown attacks {unknown}. Available: {sorted(ATTACKS)}")
    return [ATTACKS[n] for n in names]