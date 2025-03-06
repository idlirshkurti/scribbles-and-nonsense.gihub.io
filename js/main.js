// Add a blinking cursor effect to command lines
document.addEventListener('DOMContentLoaded', () => {
    const commands = document.querySelectorAll('.command');
    
    commands.forEach(command => {
        const cursor = document.createElement('span');
        cursor.classList.add('cursor');
        cursor.innerHTML = '▋';
        cursor.style.marginLeft = '5px';
        cursor.style.animation = 'blink 1s step-end infinite';
        command.appendChild(cursor);
    });

    // Add CSS for cursor animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes blink {
            from, to { opacity: 1; }
            50% { opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Add typing effect to content
    const typingElements = document.querySelectorAll('.typing-effect');
    typingElements.forEach((element, index) => {
        element.style.animationDelay = `${index * 0.1}s`;
        element.style.opacity = '1';
    });

    // Apply syntax highlighting to Python code blocks
    applySyntaxHighlighting();
});

// Function to apply syntax highlighting to code blocks
function applySyntaxHighlighting() {
    // Get all code blocks
    const codeBlocks = document.querySelectorAll('.code-block pre code');
    
    codeBlocks.forEach(codeBlock => {
        // Skip if already processed
        if (codeBlock.dataset.highlighted === 'true') return;
        
        const language = codeBlock.parentElement.parentElement.getAttribute('data-language');
        
        if (language && (language.toLowerCase() === 'python' || language.toLowerCase() === 'py')) {
            // Apply Python syntax highlighting
            highlightPythonSyntax(codeBlock);
        }
        
        // Mark as highlighted
        codeBlock.dataset.highlighted = 'true';
    });
}

// Function to highlight Python syntax
function highlightPythonSyntax(codeBlock) {
    // Get the text content
    let code = codeBlock.textContent;
    
    // Create a temporary div to safely parse HTML
    const tempDiv = document.createElement('div');
    
    // Define keywords and patterns
    const keywords = [
        'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 
        'def', 'del', 'elif', 'else', 'except', 'False', 'finally', 'for', 
        'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None', 
        'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True', 'try', 
        'while', 'with', 'yield'
    ];
    
    // Replace HTML special characters
    code = code.replace(/&/g, '&amp;')
               .replace(/</g, '&lt;')
               .replace(/>/g, '&gt;');
    
    // Pre-process triple-quoted strings (docstrings)
    // This is a simplified approach that works for most cases
    const docstringRegex = /"""[\s\S]*?"""|'''[\s\S]*?'''/g;
    const docstrings = [];
    let docstringCounter = 0;
    
    code = code.replace(docstringRegex, match => {
        const placeholder = `__DOCSTRING_${docstringCounter}__`;
        docstrings.push(match);
        docstringCounter++;
        return placeholder;
    });
    
    // Split the code into lines for easier processing
    const lines = code.split('\n');
    let highlightedCode = '';
    
    // Process each line
    for (let line of lines) {
        // Handle comments first (they take precedence)
        const commentMatch = line.match(/(#.*)$/);
        let comment = '';
        
        if (commentMatch) {
            comment = commentMatch[1];
            line = line.replace(commentMatch[1], '');
        }
        
        // Handle strings
        let inString = false;
        let stringChar = '';
        let currentToken = '';
        let processedLine = '';
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (inString) {
                currentToken += char;
                
                // Check for string end or escaped quote
                if (char === stringChar && line[i-1] !== '\\') {
                    processedLine += `<span class="string">${currentToken}</span>`;
                    inString = false;
                    currentToken = '';
                }
            } else if (char === '"' || char === "'") {
                // Process any accumulated token
                if (currentToken) {
                    processedLine += processToken(currentToken, keywords);
                    currentToken = '';
                }
                
                // Start string
                inString = true;
                stringChar = char;
                currentToken = char;
            } else {
                currentToken += char;
                
                // Check for token boundaries
                if (/[\s\(\)\[\]\{\}\:\,\.\+\-\*\/\%\=\<\>\!\;\~]/.test(char)) {
                    if (currentToken.length > 1) {
                        // Process the token except the last character
                        const tokenWithoutLast = currentToken.slice(0, -1);
                        processedLine += processToken(tokenWithoutLast, keywords);
                        currentToken = char;
                    }
                }
            }
        }
        
        // Process any remaining token
        if (currentToken) {
            processedLine += processToken(currentToken, keywords);
        }
        
        // Add back the comment
        if (comment) {
            processedLine += `<span class="comment">${comment}</span>`;
        }
        
        highlightedCode += processedLine + '\n';
    }
    
    // Replace docstring placeholders with highlighted docstrings
    for (let i = 0; i < docstrings.length; i++) {
        const placeholder = `__DOCSTRING_${i}__`;
        highlightedCode = highlightedCode.replace(
            placeholder, 
            `<span class="string">${docstrings[i]}</span>`
        );
    }
    
    // Set the highlighted code
    codeBlock.innerHTML = highlightedCode;
}

// Helper function to process a token
function processToken(token, keywords) {
    // Trim whitespace for checking but keep it in the result
    const trimmed = token.trim();
    
    // Check if it's a keyword
    if (keywords.includes(trimmed)) {
        // Replace only the keyword part, preserving whitespace
        const index = token.indexOf(trimmed);
        return token.substring(0, index) + 
               `<span class="keyword">${trimmed}</span>` + 
               token.substring(index + trimmed.length);
    }
    
    // Check if it's a decorator
    if (trimmed.startsWith('@')) {
        return `<span class="decorator">${token}</span>`;
    }
    
    // Check if it's a number
    if (/^\d+\.?\d*$/.test(trimmed)) {
        return `<span class="number">${token}</span>`;
    }
    
    // Check if it's a function call
    if (/\w+\($/.test(trimmed)) {
        const funcName = trimmed.substring(0, trimmed.length - 1);
        const index = token.lastIndexOf(funcName);
        return token.substring(0, index) + 
               `<span class="function">${funcName}</span>` + 
               token.substring(index + funcName.length);
    }
    
    // Default case
    return token;
} 