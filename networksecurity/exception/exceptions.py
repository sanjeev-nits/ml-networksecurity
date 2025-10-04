import sys
from networksecurity.logging import logger
class NetworkSecurityError(Exception):
    """Base class for all network security related exceptions."""
    def __init__(self,error_massage,error_detail:sys):
        super().__init__(error_massage)
        self.error_massage=error_massage
        _,_,ecc_tb=error_detail.exc_info()

        self.line_number=ecc_tb.tb_lineno
        self.file_name=ecc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in script: {self.file_name} at line number: {self.line_number} error massage: {self.error_massage}"
