%%
%% This is file `ORSC.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% ORSC.dtx  (with options: `cls,art')
%% 
%% This is a generated file.
%% 
%% Copyright 2005-2006
%% CTEX.ORG and any individual authors listed elsewhere in this file.
%% 
%% This file was generated from file(s) of the ORSC package project.
%% -----------------------------------------------------------------
%% 
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% This file may only be distributed together with a copy of original
%% source files. You may however distribute original source files
%% without such generated files. Copying of this file is authorized
%% only if either:
%% (1) you make absolutely no changes to your copy, including name; OR
%% (2) if you do make changes, you first rename it to some other name.
%% 
%% To produce the documentation run the original source files ending
%% with `.dtx' through LaTeX.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ORSC}
  [2010/10/18 v0.2k ORSC
   article document class]
\RequirePackage{keyval}
\def\ProcessOptionsWithKV#1{%
  \let\ORSC@tempa\@empty
  \def\ORSC@getoptionname##1=##2\@nil{##1}
  \@for\CurrentOption:=\@classoptionslist\do{%
    \@ifundefined{%
      KV@#1@\expandafter\ORSC@getoptionname\CurrentOption=\@nil
    }{}{%
      \edef\ORSC@tempa{\ORSC@tempa,\CurrentOption,}%
      \expandafter\def\csname ds@\CurrentOption\endcsname{}
    }%
  }%
  \edef\ORSC@tempa{\noexpand\setkeys{ORSC}{\ORSC@tempa}}
  \ORSC@tempa
}%
\def\ORSC@config{ORSC}
\define@key{ORSC}{config}[ORSC]{\def\ORSC@config{#1}}
\newif\ifORSC@dvips \ORSC@dvipsfalse
\DeclareOption{dvips}{\ORSC@dvipstrue}
\DeclareOption{dvipdfm}{\ORSC@dvipsfalse}
\newif\ifORSC@english \ORSC@englishfalse
\DeclareOption{english}{\ORSC@englishtrue}
\DeclareOption{chinese}{\ORSC@englishfalse}
\newif\ifORSC@print \ORSC@printtrue
\DeclareOption{print}{\ORSC@printtrue}
\DeclareOption{screen}{\ORSC@printfalse}
\newif\ifORSC@debug \ORSC@debugfalse
\DeclareOption{debug}{\ORSC@debugtrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexart}}
\ProcessOptionsWithKV{ORSC}
\ProcessOptions
\LoadClass[c5size,a4paper,sub4section,fancyhdr,psfont]{ctexart}[2004/08/14]
%% check pdfTeX mode
\RequirePackage{ifpdf}
%% math packages
\RequirePackage{amsmath,amsthm,amsfonts,amssymb,bm}
%% graphics packages
\RequirePackage{graphicx,epstopdf}
%% other packages
\RequirePackage{url}
%% hyperref package
%\ifpdf   % We're running pdfTeX in PDF mode
%  \RequirePackage[pdftex]{hyperref}
%\else    % We're not running pdfTeX, or running pdfTeX in DVI mode
%  \ifORSC@dvips
%    \RequirePackage[dvips]{hyperref}
%  \else
%    \RequirePackage[dvipdfm]{hyperref}
%  \fi
%  \AtBeginDvi{\special{pdf:tounicode GBK-EUC-UCS2}} % GBK -> Unicode
%\fi
%\hypersetup{CJKbookmarks,%
%       bookmarksnumbered,%
%colorlinks,%
%linkcolor=blue,%
%citecolor=blue,%
%              plainpages=false,%
%            pdfstartview=FitH}
%\ifORSC@print
%  \hypersetup{colorlinks=false}
%\fi
\AtEndOfPackage{\makeatletter\input{\ORSC@config.cfg}\makeatother}
%%
%% the global environments
%%
\oddsidemargin 7 true mm
\if@twoside
  \evensidemargin 7 true mm
\else
  \evensidemargin 7 true mm
\fi
\textheight 210 true mm
\textwidth 145 true mm
\headheight 15pt
\headsep 18pt
\parskip .5ex plus .5ex minus .5ex
\def\@listi{\leftmargin\leftmargini
            \topsep    6\p@ \@plus2\p@ \@minus4\p@
            \parsep    \z@
            \itemsep   \parsep}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    \z@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@ \@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \parsep}
\ifx\chinesematter\undefined
  \def\chinesematter{%
    \def\baselinestretch{1.3}\selectfont}
\fi
\ifx\englishmatter\undefined
  \def\englishmatter{%
    \def\baselinestretch{1}\selectfont}
\fi
\AtBeginDocument{
  \ifORSC@english
    \englishmatter
  \else
    \chinesematter
  \fi}
\newif\ifcollection \collectiontrue
\gdef\collection@title{}
\gdef\collection@titlemark{}
\gdef\collection@author{}
\long\def \immediate@protected@write#1#2#3{%
  \begingroup
    \let\thepage\relax
    #2%
    \let\protect\@unexpandable@protect
    \edef\reserved@a{\immediate\write#1{#3}}%
    \reserved@a
  \endgroup
  \if@nobreak\ifvmode\nobreak\fi\fi
}
\newwrite\ORSC@collection
\immediate\openout \ORSC@collection \jobname.col
\immediate\write\ORSC@collection{\space\string\clearpage}
\def\collection@write#1{%
  \immediate@protected@write\ORSC@collection
    {\let\footnote\@gobble \let\thanks\@gobble
     \let\label\@gobble \let\index\@gobble \let\glossary\@gobble}
    {#1}}
\let\ORSC@enddocumenthook\@enddocumenthook
\def\AtEndDocument{\g@addto@macro\ORSC@enddocumenthook}
\def\@enddocumenthook{\ORSC@writecol\ORSC@enddocumenthook}
\def\ORSC@writecol{
  \collection@write{\space\string\AddArticle}
  \collection@write{\space\space\space{\collection@title}}
  \collection@write{\space\space\space{\collection@titlemark}}
  \collection@write{\space\space\space{\collection@author}}
  \collection@write{\space\space\space{\jobname.pdf}}
}
\let\makecollection\relax
\renewcommand*{\title}[3][\@empty]{%
  \gdef\@title{#2\@titlenote{#3}}
  \ifcollection
    \gdef\collection@title{#2}
    \gdef\collection@titlemark{#1}
  \fi}
\newcommand*{\@titlenote}[1]{%
  \def\ORSC@title@footnote{#1}%
  \ifx \ORSC@title@footnote\@empty\else
    \footnote{#1}%
  \fi}
\gdef\@author{}
\renewcommand*{\author}[3][]{%
  \ifx \@author\@empty
    \gdef\@author{\@authorname{#2}\@authornote{#1}{#3}}
  \else
    \expandafter\gdef\expandafter\@author\expandafter
      {\@author\and\@authorname{#2}\@authornote{#1}{#3}}
  \fi
  \collection@write{\space\string\index{#2}}
  \ifcollection
    \@authorname@swap{#2}{\ORSC@authorname}
    \ifx \collection@author\@empty
      \expandafter\gdef\expandafter\collection@author\expandafter{\ORSC@authorname}
    \else
      \expandafter\expandafter\expandafter\gdef
      \expandafter\expandafter\expandafter\collection@author
      \expandafter\expandafter\expandafter{%
      \expandafter\collection@author\expandafter,\expandafter\space\ORSC@authorname}
    \fi
  \fi}
\def\@authorname#1{\@@authorname#1,\@nil}
\def\@@authorname#1,#2\@nil{%
  \def\ORSC@author@name{#2}
  \ifx \ORSC@author@name\@empty#1\else
    \@@authorname#2\@nil\space#1\fi}
\def\@authorname@swap#1#2{\@@authorname@swap#1,\@nil#2\@nil}
\def\@@authorname@swap#1,#2\@nil#3\@nil{%
  \def\ORSC@author@name{#2}
  \ifx \ORSC@author@name\@empty
    \gdef#3{#1}
  \else
    \@@authorname@swap#2\@nil#3\@nil
    \expandafter\gdef\expandafter#3\expandafter{#3 #1}
  \fi}
\def\@authornote#1#2{%
  \def\ORSC@author@address{#1}%
  \def\ORSC@author@footnote{#2}%
  \ifx \ORSC@author@address\@empty
    \ifx \ORSC@author@footnote\@empty\else
      \footnote{#2}%
    \fi
  \else
    \textsuperscript{#1}%
    \ifx \ORSC@author@footnote\@empty\else
      \textsuperscript{,}%
      \footnote{#2}%
    \fi
  \fi}
\gdef\@address{}
\newcommand*{\address}[2][]{%
  \ifx \@address\@empty
    \gdef\@address{\@addressline{#1}{#2}}
  \else
    \expandafter\gdef\expandafter\@address\expandafter
      {\@address\tabularnewline\@addressline{#1}{#2}}
  \fi}
\def\@addressline#1#2{\@addresscr\hb@xt@1ex{\textsuperscript{#1}\hss}#2}
\def\@addresscr{\def\\{\tabularnewline\@addresscr\hbox to1ex{}\ignorespaces}}
\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\hbox{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\@title\@empty
  \global\let\@author\@empty
  \global\let\@address\@empty
  \global\let\@thanks\@empty
}
\def\@maketitle{%
  \begingroup\centering%
  \let\footnote\thanks
    {\bfseries\zihao{-2} \@title \par}%
    \vskip 3ex%
    {\normalfont\normalsize\zihao{-4}
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1ex%
    {\normalfont\normalsize\kaishu\zihao{-5}
      \begin{tabular}[t]{@{}l@{}}%
        \@address
      \end{tabular}\par}%
  \endgroup}
\newenvironment{chinesetitle}
  {\chinesematter}
  {\global\collectionfalse\vskip 10pt}
\newenvironment{englishtitle}
  {\englishmatter}
  {\global\collectionfalse\vskip 10pt}
\AtBeginDocument{\null\vskip 50pt}
\def\ORSC@abstract{Abstract}
\def\ORSC@keywords{Keywords}
\renewenvironment{abstract}
  {\trivlist\itemindent\parindent\item\relax%
   \zihao{-5}\noindent%
   \textbf{\ORSC@abstract}\quad\ignorespaces}
  {\endtrivlist}
\newcommand*\keywords[1]{%
  \vskip .5ex\noindent
  \textbf{\ORSC@keywords}\quad#1\par}
\setcounter{secnumdepth}{5}
\newcommand\Ncaption[2][]{%
  \let\ORSC@caption@delimiter\CTEX@caption@delimiter
  \def\CTEX@caption@delimiter{#1}%
  \caption{#2}%
  \let\CTEX@caption@delimiter\ORSC@caption@delimiter
}
\renewcommand{\floatpagefraction}{0.80}
\newtheoremstyle{break}% name
  {}%      Space above, empty = `usual value'
  {}%      Space below
  {\itshape}% Body font
  {}%         Indent amount (empty = no indent, \parindent = para indent)
  {\bfseries}% Thm head font
  {.}%        Punctuation after thm head
  {\newline}% Space after thm head: \newline = linebreak
  {}%         Thm head spec
\newcommand*\Acknowledgements{%
  \subsection*{\ORSC@Acknowledgements}}
\renewenvironment{thebibliography}[1]
     {\section*{\refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \vskip 6\p@
      \itemsep 2\p@ \@plus\p@ \@minus\p@
      \small
      \def\baselinestretch{1}\selectfont
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\let\ORSC@save@appendix\appendix
\renewcommand*\appendix{\ORSC@save@appendix%
  \numberwithin{equation}{section}
  \numberwithin{algo}{section}
  \numberwithin{thm}{section}
  \numberwithin{defn}{section}
  \numberwithin{conj}{section}
  \numberwithin{exmp}{section}
  \numberwithin{rem}{section}
  \numberwithin{case}{section}
}%
\endinput
%%
%% End of file `ORSC.cls'.
