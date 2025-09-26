from lark import Lark
from lark.indenter import Indenter

CONFIG_PARSER = r"""


name_list: "[" NAME ("," NAME)* "]"  # [grad1, grad2...]

atom: FLOAT
      | STRING
      | function
      | "true" -> const_true
      | "false" -> const_false

function: NAME "(" [atom ("," atom)*] ")

named_arg: NAME "=" atom

arglist: atom 




UNAME: UCASE_LETTER ("_"|LETTER|DIGIT)*
LNAME: LCASE_LETTER ("_"|LETTER|DIGIT)*


%import common.FLOAT -> FLOAT
%import common.INT -> INT
%import common.ESCAPED_STRING -> STRING
%import common.CNAME -> IDENTIFIER
%import common.UCASE_LETTER
%import common.LCASE_LETTER
%import common.LETTER
%import common.DIGIT
%import common.SH_COMMENT -> COMMENT

"""
