%%
%% This was file `ecsthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%% ecsdocs.dtx  (with options: `thesis')
%%
%% Originally created by Steve R. Gunn (Copyright (C) 2001) as part of the ECSDocs class distribution
%%
%% Edited for use by Steven S. Aanen (2011)
%%
\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{mthesis.style}
              [2003/25/04 v1.6
   LaTeX document class]
\def\baseclass{book}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
  \edef\@curroptions{\@ptionlist{\@currname.\@currext}}
  \@tempswafalse
  \@tfor\@this:=#2\do{
    \@expandtwoargs\in@{,\@this,}{,\@curroptions,}
    \ifin@ \@tempswatrue \@break@tfor \fi}
  \let\@this\@empty
  \if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{10pt}{{10pt}{11pt}{12pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
\newcommand\bhrule{\typeout{------------------------------------------------------------------------------}}
\newcommand\btypeout[1]{\bhrule\typeout{\space #1}\bhrule}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\year}

\usepackage[titletoc]{appendix}
%% ----------------------------------------------------------------
% SPACES

\usepackage{setspace}
\onehalfspacing
\usepackage{enumitem}
%\setlength{\parindent}{0pt}
%\setlength{\parskip}{2.0ex plus0.5ex minus0.2ex}

%% ----------------------------------------------------------------
% CODING
\usepackage{listings}
\lstdefinestyle{mystyle}{frame=single}
%%We can define our own markup rules voor listings if we would like
\usepackage{color} %red, green, blue, yellow, cyan, magenta, black, white
\definecolor{mygreen}{RGB}{28,172,0} % color values Red, Green, Blue
\definecolor{mylilas}{RGB}{170,55,241}
\lstset{language=Matlab,%
    basicstyle=\ttfamily,
    columns=fullflexible,
    breaklines=true,%
    style=mystyle,
    morekeywords={matlab2tikz},
    keywordstyle=\color{blue},%
    morekeywords=[2]{1}, keywordstyle=[2]{\color{black}},
    identifierstyle=\color{black},%
    stringstyle=\color{mylilas},
    commentstyle=\color{mygreen},%
    showstringspaces=false,%without this there will be a symbol in the places where there is a space
}

%% ----------------------------------------------------------------
% TABLES

\setlength{\tabcolsep}{5pt}			% Set space between columns in tables (default 5pt)
\renewcommand{\arraystretch}{1.2} % Set space between rows in tables (default 1)
\usepackage{booktabs}         % Pro table layouts with correct hline spacing/weights
\usepackage{ctable}						% Nice tables
\usepackage{multirow}
\usepackage{threeparttable}
\usepackage{siunitx}
\usepackage{multicol}
\usepackage{dcolumn}          % .-aligned decimal columns
\newcolumntype{d}[1]{D{.}{.}{#1}}

\usepackage{etoolbox}% <-- for bold fonts
\newcommand{\ubold}{\fontseries{b}\selectfont}% <-- for bold fonts
\robustify\ubold% <-- for bold fonts

\let\oldtabular\tabular
\renewcommand{\tabular}{\small\oldtabular}

%% ----------------------------------------------------------------
% MARGINS

\usepackage{vmargin}
\setmarginsrb	{1.5in}		% left margin
						{0.6in}		% top margin
						{1.0in}		% right margin
						{0.8in}		% bottom margin
						{20pt}		% head height
						{0.25in}	% head sep
						{9pt}			% foot height
						{0.3in}		% foot sep
\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}

%% ----------------------------------------------------------------
% BREAKING

\doublehyphendemerits=10000	% No consecutive line hyphens.
\brokenpenalty=10000					% No broken words across columns/pages.
\widowpenalty=9999					% Almost no widows at bottom of page.
\clubpenalty=9999						% Almost no orphans at top of page.
\interfootnotelinepenalty=9999		% Almost never break footnotes.

%% ----------------------------------------------------------------
% HEADER

\usepackage{fancyhdr}
\lhead[\rm\thepage]{\fancyplain{}{\sl{\rightmark}}}
\rhead[\fancyplain{}{\sl{\leftmark}}]{\rm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0.3pt}

%% ----------------------------------------------------------------
% SECTION MARKS

\renewcommand{\chaptermark}[1]{\btypeout{\thechapter\space #1}\markboth{\@chapapp\ \thechapter\ #1}{\@chapapp\ \thechapter\ #1}}
\renewcommand{\sectionmark}[1]{}
\renewcommand{\subsectionmark}[1]{}

%% ----------------------------------------------------------------
% CLEAR DOUBLE PAGE

\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

%% ----------------------------------------------------------------
% MATH

\usepackage{amsmath,amsfonts,amssymb,amscd,amsthm,xspace}
\theoremstyle{plain}
\newtheorem{example}{Example}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{axiom}[theorem]{Axiom}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\usepackage[ruled]{algorithm2e}
\SetKwProg{Fn}{function}{}{}
% \usepackage{algorithm}
% \usepackage[noend]{algpseudocode}
% \algrenewcommand\algorithmicrequire{\textbf{Input:}}
% \algrenewcommand\algorithmicensure{\textbf{Output:}}
% \algnewcommand\algorithmicforeach{\textbf{for each}}
% \algdef{S}[FOR]{ForEach}[1]{\algorithmicforeach\ #1\ \algorithmicdo}
%\floatname{algorithm}{Procedure}			% Rename Algorithm float

%% ----------------------------------------------------------------
% CAPTIONS

\def\figurename{Fig.}
\def\tablename{Table}

\def\fnum@figure{\bf{\figurename~\thefigure}}
\def\fnum@table{\bf{\tablename~\thetable}}

%\usepackage[centerlast,small,sc]{caption2}

\usepackage[center,small]{caption}
\setlength{\captionmargin}{36pt}

\newcommand{\legend}[1]{
	\newlength{\captiontextwidth}
	\setlength{\captiontextwidth}{\textwidth}
	\addtolength{\captiontextwidth}{-2\captionmargin}
	\vspace{0.1mm}
	\hspace{2.7mm}\parbox{\captiontextwidth}{\singlespace \small (#1)}
}

%% ----------------------------------------------------------------
% SIDE-BY-SIDE FIGURES (STEREO)

\newcommand{\stereofigure}[6] {
	\newlength{\defaultcaptionmargin}
	\setlength{\defaultcaptionmargin}{\captionmargin}
	\setlength{\captionmargin}{0pt} % Disable the margin for a moment
	\begin{figure}[htb]
		\centering
		\begin{minipage}[t]{#1\textwidth}
			\centering
			\includegraphics[width=\textwidth]{#2}
			\caption{#3}
			\label{fig:#2}
		\end{minipage}
		\quad
		\begin{minipage}[t]{#4\textwidth}
			\centering
			\includegraphics[width=\textwidth]{#5}
			\caption{#6}
			\label{fig:#5}
		\end{minipage}
	\end{figure}
	\setlength{\captionmargin}{\defaultcaptionmargin} % Reset the margin
}

%% ----------------------------------------------------------------
% ENVIRONMENT REFS & SHORTCUTS

\newcommand{\fref}[1]{Figure~\ref{#1}}
\newcommand{\tref}[1]{Table~\ref{#1}}
\newcommand{\eref}[1]{Equation~\ref{#1}}
\newcommand{\cref}[1]{Chapter~\ref{#1}}
\newcommand{\sref}[1]{Section~\ref{#1}}
\newcommand{\aref}[1]{Appendix~\ref{#1}}

\newcommand{\bw}[1]{(#1 in black \& white printing)}

%% ----------------------------------------------------------------
% DEV TITLE PAGE

% todo: make optional
\usepackage{datetime}

%% ----------------------------------------------------------------
% MISC SPACING & IMPORTS

\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\dbltopfraction}{.85}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{.75}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9}

\usepackage{graphicx}         % Nice graphics

%\usepackage{layouts}           % Print all margins & layout statistics

\usepackage{epstopdf}

\usepackage[scriptsize]{subfigure}

\usepackage{rotating}

\usepackage{listings}

\usepackage{varioref}         % Refs with page no


% \usepackage{natbib}           % Use Natbib (Harvard) style for the refs

\usepackage{eurosym}

\usepackage{stackengine} % Used for matrix dimensions
\stackMath

\usepackage{chngcntr}
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}

%\usepackage{lstpatch}
%\lstset{captionpos=b,
%        frame=tb,
%        basicstyle=\scriptsize\ttfamily,
%        showstringspaces=false,
%        keepspaces=true}
%\lstdefinestyle{matlab} {
%        language=Matlab,
%        keywordstyle=\color{blue},
%        commentstyle=\color[rgb]{0.13,0.55,0.13}\em,
%        stringstyle=\color[rgb]{0.7,0,0} }

%% ----------------------------------------------------------------
% COLORS, PDF & LINKS

\usepackage{color}
\definecolor{cite}{RGB}{136,102,0}
% \definecolor{cite}{RGB}{0,0,0}
\definecolor{link}{RGB}{0,0,0}
%\definecolor{link}{gray}{0.6}
\definecolor{url}{RGB}{83,83,176}
%\definecolor{url}{RGB}{120,20,20}

\usepackage[pdfpagemode={UseOutlines},bookmarks=true,bookmarksopen=true,
   bookmarksopenlevel=0,bookmarksnumbered=true,hypertexnames=false,
   colorlinks,linkcolor={link},citecolor={cite},urlcolor={url},
   pdfstartview={FitV},unicode,breaklinks=true]{hyperref}
\pdfstringdefDisableCommands{
   \let\\\space
}

%% ----------------------------------------------------------------
% META-DATA

\let\origref\ref
\def\ref#1{\origref{#1}}

\newcommand*{\supervisor}[1]{\def\supname{#1}}
\newcommand*{\coreader}[1]{\def\coreadname{#1}}
\newcommand*{\examiner}[1]{\def\examname{#1}}
\newcommand*{\degree}[1]{\def\degreename{#1}}
\newcommand*{\authors}[1]{\def\authornames{#1}}
\newcommand*{\addresses}[1]{\def\addressnames{#1}}
\newcommand*{\university}[1]{\def\univname{#1}}
\newcommand*{\UNIVERSITY}[1]{\def\UNIVNAME{#1}}
\newcommand*{\studentnumber}[1]{\def\studentid{#1}}
\newcommand*{\department}[1]{\def\deptname{#1}}
\newcommand*{\DEPARTMENT}[1]{\def\DEPTNAME{#1}}
\newcommand*{\group}[1]{\def\groupname{#1}}
\newcommand*{\GROUP}[1]{\def\GROUPNAME{#1}}
\newcommand*{\faculty}[1]{\def\facname{#1}}
\newcommand*{\FACULTY}[1]{\def\FACNAME{#1}}
\newcommand*{\subject}[1]{\def\subjectname{#1}}
\newcommand*{\keywords}[1]{\def\keywordnames{#1}}
\newcommand*{\logo}[2]{\def\logofig{#1}\def\logowidth{#2}}

\supervisor		{Dr. Flavius Fr\u{a}sincar}
\coreader			{TBD}
\examiner			{}
\degree				{Bachelor of Science$^2$ in Econometrics \& Economics}
\authors			{}
\university		{{Erasmus Universiteit Rotterdam}}
\UNIVERSITY		{{EUR}}
\studentnumber{500202}
\department		{Business Analytics \& Quantitative Marketing}
\DEPARTMENT	{Econometrics Institute}
\group				{\texorpdfstring{\href{http://www.isis.ecs.soton.ac.uk}
						{Image, Speech and Intelligent Systems Group}}
						{Image, Speech and Intelligent Systems Group}}
\GROUP			{\texorpdfstring{\href{http://www.isis.ecs.soton.ac.uk}
						{IMAGE, SPEECH AND INTELLIGENT SYSTEMS GROUP}}
						{IMAGE, SPEECH AND INTELLIGENT SYSTEMS GROUP}}
\faculty				{Erasmus School of Economics}
\FACULTY			{ESE}
\addresses		{}
\subject			{}
\keywords			{}

%% ----------------------------------------------------------------
% TITLE PAGE

\usepackage{soul}		% Small capitals & letter spacing
\soulregister{\univname}{0}

\renewcommand\maketitle{
  \btypeout{Title Page}
  \hypersetup{pdftitle={\@title}}
  \hypersetup{pdfsubject=\subjectname}
  \hypersetup{pdfauthor=\authornames}
  \hypersetup{pdfkeywords=\keywordnames}
  \thispagestyle{empty}
  \begin{titlepage}
%    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \setcounter{footnote}{0}
%    \null\vfil
%    \vskip 10\p@
    \begin{center}
      \setlength{\parskip}{0pt}
      {\includegraphics[width=\logowidth, keepaspectratio=true]{\logofig} \par}
      \vskip 75\p@
      {\huge \bf \@title \par}
%      \vfill
      \vskip 25\p@
      {\normalsize by \par}
      \vskip 25\p@
      {\Large \bf \authornames \par}
      \bigskip
      {\normalsize Student ID: \studentid \par}
      {\normalsize Thesis Supervisor: \supname \par}
      {\normalsize Second Assessor: \coreadname \par}

%      {\tiny non-final version, compiled at \number\day \space \today}

      \vfill
      {\normalsize A thesis submitted in partial fulfilment for the degree of \par}
      \smallskip
      {\normalsize\caps \degreename \par}
      \bigskip
      {\normalsize at the \par}
      \bigskip
      {\normalsize \facname \par}
      \smallskip
      {\caps{\normalsize \univname} \par}
      \smallskip
      {\normalsize \deptname \par}

%      {\large \deptname \par}
      \bigskip
      \bigskip
      \bigskip
      {\small \bf \@date \par}
    \end{center}
    \par
    \@thanks
    \vfil\null
  \end{titlepage}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \cleardoublepage
}

%% ----------------------------------------------------------------
% ABSTRACT

\newenvironment{abstract}
{
  \btypeout{Abstract Page}
  \thispagestyle{empty}
  \null\vfil\vfil\vfil
  \begin{center}
    \setlength{\parskip}{0pt}
    {\bf \flushleft \LARGE {Abstract} \par}
  \end{center}
}
{
  \vfil\vfil\vfil\null
  \cleardoublepage
}
%% ----------------------------------------------------------------
% TOC

\setcounter{tocdepth}{2}

\addtocounter{secnumdepth}{1}
\newcounter{dummy}
\newcommand\addtotoc[1]{
  \refstepcounter{dummy}
  \addcontentsline{toc}{chapter}{#1}
}

\renewcommand\tableofcontents{
\btypeout{Table of Contents}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi
   \cleardoublepage
}\end{spacing}
}

%% ----------------------------------------------------------------
% LIST OF FIGURES

\renewcommand\listoffigures{
\btypeout{List of Figures}
\addcontentsline{toc}{chapter}{\listfigurename}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}
              {\MakeUppercase\listfigurename}}
    \@starttoc{lof}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}

%% ----------------------------------------------------------------
% LIST OF TABLES

\renewcommand\listoftables{
\addcontentsline{toc}{chapter}{\listtablename}
\btypeout{List of Tables}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}
    \@starttoc{lot}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}

%% ----------------------------------------------------------------
% NOMENCLATURE

\newcommand\listsymbolname{Nomenclature}
\usepackage{longtable}
\newcommand\listofsymbols[2]{
\btypeout{\listsymbolname}
\addtotoc{\listsymbolname}
    \chapter*{\listsymbolname
      \@mkboth{
          \MakeUppercase\listsymbolname}{\MakeUppercase\listsymbolname}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    \cleardoublepage
}

%% ----------------------------------------------------------------
% ACKNOWLEDGEMENTS

\newcommand\acknowledgements[1]{
\btypeout{Acknowledgements}
\addtotoc{Acknowledgements}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large\bf Acknowledgements \par}\end{center}
{\normalsize #1}
\vfil\vfil\null
\cleardoublepage
}

%% ----------------------------------------------------------------
% DEDICATORY

\newcommand\dedicatory[1]{
\btypeout{Dedicatory}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large \sl #1}\end{center}
\vfil\null
\cleardoublepage
}

\renewcommand\backmatter{
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \addtotoc{\bibname}
  \btypeout{\bibname}
  \@mainmatterfalse}
\endinput

%% ----------------------------------------------------------------
%% EOF
